# Erratum: réécriture in-place à 225957b

Le fichier `artifacts/v1/null-source-records.jsonl` a été RÉÉCRIT in-place
à 225957b (au lieu d'un chemin neuf append-only). La version précédente
(6bbe858) était déjà correcte (74.6% invalid labels), mais la pratique
de réécriture viole la discipline append-only.

**Cause racine**: le générateur null_source.py était inline (heredoc) dans
le terminal au moment de 6bbe858; à 225957b le générateur a été commité
comme fichier .py et les records régénérés — sur le MÊME chemin au lieu
d'un chemin neuf. Les données sont identiques dans les deux versions
(sha blob 87aafc3b…), mais la traçabilité append-only a été brisée.

**Leçon**: toujours écrire sur un chemin neuf; jamais réécrire un artefact
publié, même si le contenu est inchangé.

**Note pour version future du générateur**: neutraliser le champ
`execution.observable_result` (vestigial, deep-copy du canon — source de
confusion d'audit tagi-5 15:29 car il porte "valid" alors que le label
null est uniform-tous).
