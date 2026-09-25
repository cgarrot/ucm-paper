# VISION — pourquoi ce projet existe

**Date :** 23 septembre 2026 · **Auteur :** cgarrot (fondateur), retranscrit par l'orchestrateur
**Statut :** document fondateur — la motivation produit qui précède et encadre la [spec](../universal-control-model-project-spec.md). En cas de conflit sur la méthode, la spec gagne ; en cas de doute sur la direction, ce document gagne.

---

## 1. L'histoire de départ

Tout commence avec deux outils qui marchent très bien :

- **Muse Voice Transcripts** — transcription vocale en direct, excellente
- **Jev** — le nouveau modèle de TypeSafe (catégorie « System One Models », sept. 2026), qui comprend les demandes

Le constat fondateur : **Jev comprend, mais n'arrive pas à réaliser les tâches complexes.** Il a du mal à décomposer, et exécuter étape par étape avec un gros modèle à chaque décision est **beaucoup trop lent**.

L'intuition : **un petit modèle dédié à l'exécution serait beaucoup plus efficace.** Ce projet est né pour construire cette pièce — avec la rigueur nécessaire pour qu'on puisse lui faire confiance.

## 2. La stack cible

```
VOIX ──► Muse Voice Transcripts (existe, marche très bien)
              │ texte
              ▼
         Jev / gros modèle : COMPREND + DÉCOMPOSE
              │ objectif structuré (genres d'interaction)
              ▼
         UCM : EXÉCUTE — le petit modèle (≈1 M params, ~1 ms/décision)
              │ s'adapte (in-context), se récupère (P3), escalade si bloqué
              ▼
         RÉSULTAT mesuré → si échec : re-décomposition ciblée
```

**UCM = la couche d'exécution.** Le gros modèle lit et décide quoi faire ; le petit fait — vite, bien, bon marché, et il sait dire « je ne sais pas ».

## 3. Les principes qui découlent de cette vision

1. **La compacité est une contrainte mesurée, pas un dogme** (2-3 M params acceptables) — ce qui compte : le coût/latence/fiabilité par action, face à l'alternative « gros modèle à chaque décision »
2. **Apprendre l'ordre**, pas seulement scorer des options — c'est la différence avec les scoreurs one-pass (cua-s1-forms code l'ordre en dur)
3. **Interfaces décrites, pas pixels** — le petit modèle lit une description structurée du monde (entités, relations, genres, candidats) ; le pont vers le vrai logiciel est un compilateur (DOM→genres), pas un changement de modèle
4. **Chaque capacité doit être prouvée** (gates scellés, contrôles, IC) — un agent non mesuré est un agent non livrable
5. **Le gros modèle reste hors du chemin rapide** — il intervient pour comprendre, décomposer, débloquer ; jamais à chaque clic

## 4. Où on en est (23/09/2026)

| Pièce | État |
|---|---|
| Muse Voice Transcripts | ✅ existe |
| Jev (comprendre) | ✅ existe |
| **UCM au laboratoire** (mondes jouets, oracle exact) | ✅ V0 prouvée : 695k params, 97,5 % de généralisation, 99,4 % de recombinaison, ~1 ms/décision |
| UCM profondeur (tâches >~15 étapes) | 🔬 P1 (calcul itératif) |
| UCM multi-mondes + adaptation in-context | 🔬 P2 (mondes procéduraux à genres) |
| Compilateur vrai logiciel → genres | ⬜ étape Web (Jev peut y contribuer) |
| Démo bout en bout voix→Jev→UCM | ⬜ S5 |

## 5. Ce que ce projet ne prétend pas

- Pas un concurrent des gros modèles généralistes (Gato, LLM) — c'est une **couche**, pas un cerveau
- Pas « l'AGI » — un exécutant fiable, mesuré, bon marché
- Pas une révolution — une pièce manquante, construite proprement

## 6. Filiation des documents

- Vision (ce document) → direction
- [Spec](../universal-control-model-project-spec.md) → protocole scientifique
- [PLAN.md](PLAN.md) → exécution & leçons
- [REPORT-V0.md](REPORT-V0.md) → première preuve
- `reports/web-watch-2026/` → veille & positionnement 2026 (la catégorie « System One Models » et cua-s1-forms 706k y sont documentées)
