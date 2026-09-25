# PLAN — génération et scellement test2 V1-bis (pré-enregistrement Stage-B)

**Statut : PRÉ-ENREGISTREMENT avant génération — aucune donnée test2 vue.** Complète le protocole `a1a8b9d` (§5 étape 4) dont l’exécution attend le GO lead explicite. Instrument VERT COMPLET (`6957238`). HOLD test2 inchangé jusqu’au GO. Append-only.

## 1. Seeds et iso-disjonction

- **Seed de génération test2 = `20261003`** (réservé depuis l’origine, jamais consommé — attesté par D9/incidents) ; **remplacement unique = `20261004`** (même statut), règle §3-bis du protocole (un remplacement max, échec persistant ⇒ NO-GO technique documenté).
- **Pools iso-disjoints** : les layouts test2 doivent être disjoints des layouts d’adaptation (réservoir des 10 fichiers `v1bis-gen-*`, gate `24ad755`) ET des pools TGK canon. Vérification M1-style : recomputation `layout_hash` depuis le spec embarqué, chevauchement ⇒ rejet fail-closed avant scellement.

## 2. Taille et équilibrage (déclarés avant génération)

- **600 épisodes, 150 par prédicat** (VIEW/SET/CHOOSE/SUBMITTED) — critère M1 de l’instrument audité ; PAS de quota par layout (comptes variables licites si équilibre par prédicat).
- Épisodes auto-suffisants (spec de layout embarqué, task, d_star), schéma compatible eval closed-loop §9.4 (horizon 64, STOP natif).

## 3. Procédure de scellement (Stage-B)

1. Génération par le générateur committé (producer commit pinné), fichier `artifacts/test2/v1bis-test2-20261003.episodes.jsonl` (chemin neuf).
2. **Vérifications avant scellement** (fail-closed) : M1 (600/150×4, hashes recomputés), iso-disjonction §1, canonicalité JSONL (writer RR non requis pour les épisodes — ordre de génération natif conservé).
3. **Data-manifest Stage-B** : SHA-256 complet du fichier publié AVANT le run, lié au freeze par le slot eval (le GO retire le Stage-B LOCK et lie le hash attendu).
4. **Une seule lecture** au run (`SealedOpenRegistry`, hash attendu vérifié dans le flux) ; toute divergence ⇒ abort sans consommation.

## 4. GO étape 4 — conditions (toutes requises)

- [x] Instrument VERT COMPLET (`6957238`, PASS pinné).
- [ ] Audit PASS de CE plan (tagi-5, octets pinnés).
- [ ] Budget OPS confirmé (estimation : ~120 cellules d’adaptation × 2000 updates + eval 600×120 — CPU-only d’après profil `8954200`, ordre de grandeur 5-15 h ; confirmation tagi-4 requise).
- [ ] Vérification finale lead : seeds 20261003/04 jamais consommés (re-scan artefacts).

Le GO lève : eval Stage-B LOCK + émet `V1BIS_STEP4_GO=LEAD_APPROVED`. **Sans GO : rien n’est généré.**
