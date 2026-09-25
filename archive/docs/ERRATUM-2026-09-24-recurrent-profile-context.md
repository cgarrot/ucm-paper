# ERRATUM 2026-09-24 — Contexte de mesure du profil S2b (RecurrentRefine T=32)

- **Statut** : erratum séparé, append-only. Le rapport `reports/recurrent-s2b-profile.json` (`472348c`, blob `b9364e4f…`) et ses logs **ne sont pas modifiés**. Précisions demandées par l'audit pinné tagi-5 (`m_muffpjbf_9b2f1e1c`).
- **Objet** : lever deux ambiguïtés d'interprétation du profil récurrent (notes mineures (a) et (b) de l'audit).

## (a) Journal de charge — WT vs blob committé

- Le `load_logger` échantillonnait encore pendant/après le commit → le **worktree** de `reports/recurrent-s2b-load.jsonl` a continué de croître (`M` constaté par l'auditeur).
- **Référence auditable = le blob committé** `255bd24fc87ce5402723a292379d25b2b48f6f6ef1d7721fb5b048182f15de56` (`git show HEAD:… | sha256` vérifié).
- **Nettoyage effectué** : logger arrêté (PID 69765), WT restauré au blob committé (`git checkout --`), statut propre. Les échantillons post-commit (période inactive) n'ont pas été conservés.

## (b) p95 « full décision » (11,789 ms) < p95 « model-only » (14,084 ms) — contexte

Ce n'est **pas** une contradiction : les segments sont **indépendants** (§10.2) et mesurés **séquentiellement** dans le même processus enfant :

1. **Ordre** : segment *model-only* exécuté **en premier** (juste après le warm-up de 50), segment *full* **en second**.
2. **Dérive de charge** : le processus enfant a tourné pendant la **décroissance de charge** post-entraînement rec (échantillons du journal : **4,94 → 6,08** sur la fenêtre de mesure ; le training rec venait de terminer). Un segment plus tardif bénéficie d'un CPU moins disputé — d'où un p95 *full* inférieur au p95 *model-only*.
3. Les deux segments cyclent les **mêmes 32 observations** ; la différence est temporelle (ordre/charge), pas structurelle.
4. **Conséquence pratique** : c'est le p95 du segment **model-only** (14,084 ms CPU) qui est la mesure conservative utilisée pour le gate ≤ 20 ms — le chiffre *full* (11,789) ne le remplace pas.
5. **Amélioration notée pour les prochains profils** : horodater chaque segment et publier l'ordre des segments (le template `profile_*.py` sera ajusté).

## (c) Ratios à citer

- Per-update rec/B144 : **p50 mesuré = 11,911×** ; **moyenne soutenue ≈ 10,1×** (recalcul audit tagi-5) — citer les deux.
- RSS : **9 721 MB** (rec) vs **1 437 MB** (B144) ⇒ borne des workers du planning factoriel (déjà intégré).

---

*Rédigé par tagi-4 (OPS), 2026-09-24, sur notes d'audit tagi-5 `m_muffpjbf_9b2f1e1c`. SHA de cet erratum : publié hors fichier.*
