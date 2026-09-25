# ERRATUM — protocole V1-bis v2 (`a1a8b9d`) — hyperparamètres d'adaptation

**Statut : erratum append-only AVANT tout run V1-bis.** Le protocole `docs/PROTOCOL-V1BIS-v2-2026-09-23.md` (commit `a1a8b9d`, SHA `53771350…`) §2 exige « même architecture, updates, batch, optimiseur » pour tous les bras **sans fixer les valeurs** — une omission détectée lors de l'audit du freeze (`4ef98ca`, tagi-5 15:38). Aucune donnée d'entraînement/évaluation V1-bis n'ayant été vue, ces valeurs sont fixées **maintenant** comme préenregistrées :

- **updates d'adaptation = 2000** (par cellule bras×seed×k ; valeur de l'instrument M-V1b, reprise à l'identique) ;
- **batch = 64** ; optimiseur AdamW et règle de checkpoint (`best_validation` sur split de validation de l'adaptation) inchangés de l'instrument audité.

Ces valeurs lient le freeze officiel (champ `protocol.updates` = 2000) ; toute divergence future = nouveau document. Le présent erratum ne modifie ni l'estimand, ni le k_plan [64, 128, 256] épisodes, ni les seuils, ni la séquence du protocole. **HOLD test2 inchangé.**
