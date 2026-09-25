# PC XMG (RTX 3070) — guide d'accès et d'utilisation

**Cible :** agents travaillant sur UCM (room mesh `try-agi`).  
**Rôle :** c'est le « matériel secondaire optionnel » de la [spec](../universal-control-model-project-spec.md) — **RTX 3070 laptop 8 Go, seulement si un besoin concret justifie un second backend**. Ce n'est pas le backend principal (M5/MLX).

---

## 1. Accès SSH (vérifié et fonctionnel le 22 sept. 2026)

```bash
ssh cgarrot@192.168.1.58     # ou: ssh xmg (alias ajouté dans ~/.ssh/config du Mac)
```

- **Authentification par clé, sans mot de passe** (BatchMode OK — utilisable par un agent directement).
- Utilisateur : `cgarrot`. `root` n'est pas accessible en SSH.
- Machine : `xmg-fusion-m22`. Si `192.168.1.58` ne répond plus → `nc -z -G 1 <ip> 22` pour retrouver le PC sur `192.168.1.0/24` (IP possibly renouvelée par DHCP ; le hostname `xmg-fusion-m22` fait foi).

## 2. Matériel

| | |
|---|---|
| CPU | i7-11800H, 16 cœurs |
| GPU | **NVIDIA RTX 3070 Laptop, 8 192 MiB** (driver 595.91.07, CUDA driver 13.2) |
| RAM | 30 Go |
| OS | Debian 13 (trixie), kernel 6.12 |
| Disque | 363 Go, **88 % plein — 42 Go libres** ⚠️ |
| Outils | `tmux`, `git`, `rsync`, `python3 -m pip` |

## 3. Ce qui est déjà installé

- **Python 3.13.5** (système, `/usr/bin/python3`), pip en mode user (`~/.local`).
- **PyTorch 2.5.1+cu121 fonctionne out-of-the-box**, CUDA dispo (`torch.cuda.is_available() == True`). Test en une commande :

```bash
ssh cgarrot@192.168.1.58 'python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"'
```

- CUDA toolkit 13.3 dans `/usr/local/cuda*` (`nvcc` pas dans le PATH d'un shell non-login → `source ~/.zshrc` ou exporter le PATH si besoin de compiler).
- Pas de docker, pas de conda/uv. ComfyUI et d'autres projets GPU vivent déjà sur le disque (home encombré → nettoyer avant tout gros dataset).

## 4. Mini-setup pour utiliser le GPU

**a) Vérifier que la carte est libre avant de lancer :**

```bash
ssh xmg 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

(0 % et ~66 MiB = idle. Si quelqu'un tourne déjà → se coordonner sur le mesh.)

**b) Envoyer du code / des données (rsync, reprise possible) :**

```bash
rsync -avz --progress ./dossier/ xmg:~/ucm/dossier/
```

**c) Isoler l'environnement du projet (venv standard) :**

```bash
ssh xmg
python3 -m venv ~/ucm/.venv
source ~/ucm/.venv/bin/activate
pip install torch numpy …   # le torch système marche déjà ; le venv évite de casser les autres projets
```

**d) Lancer un entraînement dans tmux** (survit à la déconnexion SSH) :

```bash
tmux new -s ucm-train
source ~/ucm/.venv/bin/activate && python train.py …
# détacher: Ctrl-b d — reprendre: tmux attach -t ucm-train
```

**e) Surveiller :**

```bash
ssh xmg 'nvidia-smi'                       # charge/temp/mémoire GPU
ssh xmg 'tmux ls'                          # sessions en cours
```

Laptop : cap 80 W, throttling thermique possible — prévoir du profiling soutenu (la spec §10.2 exige de détecter le throttling sur un segment long).

## 5. Règles UCM spécifiques (spec §10.3, §10.2)

1. **La RTX 3070 ne fait pas partie du protocole principal.** Toute migration M5→RTX exige son propre profiling et des résultats publiés séparément.
2. **Pas d'attribution à l'architecture d'un gain venant du backend ou de la précision** (cu121 vs MLX, FP16 vs FP32 : documenter).
3. **Journaliser chaque run** dans `artifacts/` du repo (durée, machine, précision, batch, mémoire, temp/throttle) — l'enveloppe V0 de 24 h d'accélérateur est cumulée, tous backends confondus.
4. Entraînements longs → toujours en `tmux`, jamais en session SSH directe.
5. **Règle ops (tagi-4) — aucun sync d'artifacts sans vérif espace :** tout `rsync`/transfert d'artifacts, datasets ou checkpoints vers le XMG exige un `df -h /` préalable. **État au 22/09/2026 : 363 Go, 88 % utilisés, 42 Go libres.** Si l'espace est insuffisant : nettoyer uniquement des dossiers temporaires créés par le projet, jamais les projets existants du home. Le transfert du repo UCM seul (hors `.venv`/`artifacts`) ≈1 Mo.

## 6. Pièges connus

- **Disque à 88 %** (42 Go libres au 22/09/2026) : `df -h /` obligatoire avant tout transfert (règle §5.5).
- Le shell distant est **zsh** ; en SSH non-interactif, `nvcc`/chemins CUDA ne sont pas dans le PATH (voir §3).
- `192.168.1.46` (OpenSSH 6.6) est un autre appareil du réseau — ce n'est **pas** le XMG, ne pas s'y tromper.
- Clé utilisée depuis ce Mac : celle par défaut (`~/.ssh/id_ed25519` ou agent) — pas de passphrase demandée.
