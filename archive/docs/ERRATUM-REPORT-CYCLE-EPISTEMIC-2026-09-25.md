# ERRATUM — rapport cycle S2b→P2 : distinction épistémique intra-bande vs strict

**Statut : erratum append-only.** Le rapport `REPORT-CYCLE-S2B-P2-FINAL.md` (`a1f4d0e`) doit distinguer les statuts épistémiques des deux verdicts de la Recherche B — la formulation corrigée ci-dessous prévaut pour toute publication.

## Correction

Le **GO intra-bande (97,0 %)** est **préenregistré** (design gelé `5290643` avec sa contrainte §5.4 « composantes constructibles » — la sémantique même du test v01). Le **KILL strict (0,0 %)** est **exploratoire** : le test strict (actions optimales jamais optimales dans le canon) est l'exact contraire de la contrainte §5.4 du design gelé — il a été ajouté **après** le gel ; le champ `design_frozen: 5290643` de la v02 est une référence périmée pour ce sous-test.

**Formulation prévalente** : « intra-bande : généralisation aux nouveaux layouts établie (préenregistré, GO) ; strict inter-signatures : transfert compositionnel NON établi (test **exploratoire**, KILL) ; la calibration générale ne suffit pas pour la composition — prochain verrou scientifique. »

## Traces

- **REACH_long_door absent** du strict (60/60 = AT_long_door) : s'explique probablement par la §5.4 elle-même — la composition REACH_long_door devait être vide/non constructible dans la famille tenue à l'écart (comme DROP+REACH en P2) et fut sautée par le générateur. À confirmer côté générateur (tagi-2). Le KILL n'en est pas affecté.
- Le KILL strict reste **honnête et informatif** (0,0 % sur 5 seeds, contrôles inclus) — mais ce n'est pas un critère gelé qui a parlé ; tout futur test compositionnel devra être **préenregistré avec sa propre contrainte de génération** (l'inverse de §5.4).

## Leçon au registre

Un test ajouté après le gel du design ne peut pas citer le design gelé comme préenregistrement — il est exploratoire par construction et doit être étiqueté tel.
