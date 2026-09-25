# DESIGN — Parallélisation bras×seeds du runner §12 (revue tagi-2 AVANT implémentation)

**Statut:** REVU — Option B GO sous 4 conditions (tagi-2 21:19, m_mud258ce).
Répartition: tagi-2 fournit le prototype de fusion scellée (B3+B4) dans son
couloir; WS-C implémente les writers (post-run-only). L'option A est REJETÉE:
même la variante replay-canonique laissait une fenêtre d'interleaving live
non-déterministe sous lock — défaut fatal pour un harnais de mesure scellé.

## Conditions de l'Option B (tagi-2, contraignantes)

- **B1 CHAÎNE:** chaîne sha256 indépendante PAR WORKER (mécanique actuelle
  inchangée, genesis par worker).
- **B2 ORDRE TOTAL DÉTERMINISTE:** fusion reconstruite par règle de CONTENU
  (arm, k, worker_id, seq) — le champ wall-clock "t" reste dans le payload
  scellé comme métadonnée humaine mais est EXCLU de toute décision d'ordre.
- **B3 FUSION QUI LIE TOUT:** le merge artifact scelle la LISTE TRIÉE des
  workers [(worker_id, count, head_seal, sha256_entries)] + la règle d'ordre
  + le hash de la séquence fusionnée. Silencer un worker entier (drop de
  fichier) = rupture du seal de fusion → RuntimeError. (Piège couvert: une
  chaîne par-worker seule ne détecte PAS l'absence d'un worker.)
- **B4 IDMMPOTENCE:** merge déterministe pur (mêmes inputs worker → même
  seal de fusion) → vérifiable byte-identical sur XMG.
- **VERSIONNEMENT:** "log_version": 1 (séquentiel) / 2 (parallèle); le run
  EN COURS (v1) reste lisible inchangé; verify() dispatche sur la version.
- **Prototype:** interactions-log-w{N}.json par worker + interactions-merge.json
  scellé (seal_manifest; timing exclu des ordres — leçon #21).
**Contraintes:** le run M-V1b séquentiel en cours est la référence; toute
parallélisation future doit produire des nombres ÉQUIVALENTS, pas seulement
rapides; le log d'interactions doit rester infalsifiable après fusion.

## 1. Graine de parallélisation

Unité de travail = CELLULE (arm, seed, k) — 75 cellules au complet. Chaque
cellule est INDÉPENDANTE par construction (init fraîche partagée par seed
reconstituée localement via mx.random.seed(9000+seed) — déterministe, donc
reproductible par-worker sans communication). Aucun état partagé entre
cellules SAUF: (i) l'ouverture unique du test scellé, (ii) le log
d'interactions, (iii) l'artefact final.

Pool de workers (ProcessPool, n≤4 pour rester léger pendant d'autres runs):
chaque worker reçoit (arm, seed, k, chemins scellés, seed d'init 9000+seed)
et rend {cellule: summary, résultats par épisode, couverture locale, wall_s,
hashes locaux}.

## 2. Log d'interactions — DEUX options (décision tagi-2)

### Option A: écrivain unique (recommandée par défaut)
Le processus père est le SEUL écrivain du InteractionsLog. Les workers ne
touchent JAMAIS le log; ils déclarent des ÉVÉNEMENTS dans leur résultat
(liste d'événements signés par worker), et le père les rejoue À LA FIN dans
l'ordre de complétion, scellant la chaîne sha256 comme aujourd'hui.
- Avantage: une seule chaîne, ordre total, tamper-evidence inchangée,
  re-vérification actuelle fonctionne tel quel.
- Inconvénient: les horodatages des événements workers sont post-hoc (temps
  de complétion, pas d'émission) — documenté dans chaque entrée
  ("replayed_from_worker").

### Option B: logs par-worker fusionnés + scellés
Chaque worker écrit son propre journal scellé (chaîne sha256 locale, gène
dérivée de (run_id, worker_id) publiée par le père AVANT dispatch). Fusion
post-run: concaténation ordonnée (worker_id, seq), puis CHAÎNE DE SCELLEMENT
DE TÊTE (head-seal): sha256 sur la liste des sceaux finaux de chaque worker,
écrite par le père. Falsifier un worker casse sa chaîne locale; falsifier la
fusion casse le head-seal; réordonner des workers casse l'ordre publié.
- Avantage: horodatages d'émission réels, pas de point d'écriture unique
  (pas de goulot).
- Inconvénient: la vérification devient deux niveaux (locale + tête); le
  code de vérification actuel doit être étendu (et re-audité).

**Recommandation WS-C: Option A** (simplicité d'audit > fidélité
d'horodatage; les événements pertinents — ouverture du test, hashes scellés
— sont TOUS émis par le père de toute façon). L'option B se justifie si des
événements worker-side devaient compter (ex: interactions cible directes
par worker, aujourd'hui inexistantes).

## 3. Équivalence numérique

- CPU: chaque worker a son propre état MLX; les cellules étant
  indépendantes et seedées, les résultats sont BITWISE identiques au
  séquentiel (même device, même précision). Test de non-régression: une
  cellule exécutée séquentiellement vs en pool → summary identiques.
- L'ÉVALUATION par cellule reste séquentielle en interne (le batched
  rollout (a) est une optimisation orthogonale, combinable après re-test
  d'équivalence).

## 4. Ce qui ne change PAS

- Ouverture unique du test: le PÈRE matérialise les épisodes/records, calcule
  et logge les hashes de contenu, PUIS dispatche les données (spécifications
  sérialisées) aux workers. Les workers ne lisent jamais les fichiers
  scellés directement (single-read au niveau processus).
- Coverage: fusion des dictionnaires par-worker (clés disjointes par
  construction arm@k — vérifié par assertion).
- Checkpoint final à budget fixe: inchangé (chaque worker sauvegarde sa
  cellule; le père agrège les chemins).

## 5. Points de revue demandés à tagi-2

1. Option A vs B (§2) — et si B: relecture du schéma de head-seal.
2. Le "single-read au niveau processus" (§4) vous paraît-il conforme à
   l'exigence d'ouverture unique, ou faut-il un gardien de fichier
   (open-once FD partagé)?
3. Ordre de replay des événements workers (complétion vs (arm,seed,k)
   canonique) — recommandation: canonique pour la reproductibilité du log.
