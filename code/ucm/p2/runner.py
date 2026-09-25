"""P2-1 (E) — Runner: 3 bras × 8 seeds × 2 familles, O_EXCL par cellule,
checkpoints + raws, budget journalisé, gardes standard.

Cellule = (famille, bras, seed): entraîne P2Model frais (perte conjointe
policy + λ·effet) sur les records train de la famille avec le contexte du
bras; évalue closed-loop sur les tâches TEST (contexte du bras au moment de
la décision). Persistance O_EXCL par cellule dès le premier commit.
"""
from __future__ import annotations

import json
import os
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASET = os.path.join(_REPO, "artifacts/p2-dataset.json")
_OUT = os.path.join(_REPO, "artifacts/p2-run")

ARMS = ("D0", "D1-mixed", "D1-shuffled")
SEEDS = tuple(range(8))
UPDATES = 4000         # gelé par sonde de convergence 21:11 (policy converge à 0.0019; le plateau du total = CE de la tête)
EVAL_SEED = 60000


def load_dataset(path=_DATASET):
    return json.load(open(path, encoding="utf-8"))


def _examples(model, records):
    import numpy as np
    from ucm.p2.tensorize_p2 import tensorize_obs_p2
    from ucm.model.tensorize import labels_from_supervision
    exs = []
    for rec in records:
        ex = tensorize_obs_p2(rec["policy_input"])
        ex["labels"] = labels_from_supervision(rec["policy_input"],
                                               rec["supervision"]["optimal_actions"])
        ex["effect_labels"] = np.asarray(rec.get("effect_labels", [0] * len(
            rec["policy_input"]["candidates"])), dtype=np.int32)
        exs.append(ex)
    return exs


def _collate_p2(exs):
    """Collate P2 = collate NATIF exact (slots _slot_tables — bug 23:44: la
    version maison rassemblait les messages ÉMIS au lieu des REÇUS: le sens
    natif est slot 2e → receveur=subj, msg_other=obj; mon grep par
    msg_other==j inversait la direction d'agrégation), avec D_IN_P2."""
    import ucm.model.tensorize as _T
    from ucm.p2.tensorize_p2 import D_IN_P2
    old = _T.D_IN
    _T.D_IN = D_IN_P2
    try:
        return _T.collate(exs)
    finally:
        _T.D_IN = old


def train_cell(model, records, updates, seed, labels_fn=None):
    import random
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import AdamW
    from mlx.utils import tree_map
    from ucm.model.loss import set_bc_loss
    import numpy as np
    # HOOK labels_fn (tagi-2 ablation contrôlée, 12:26): callable(
    # effect_labels, record) -> effect_labels override, appliqué par record
    # AVANT la perte; None = identité — pipeline inchangé par défaut.
    if labels_fn is None:
        labels_fn = lambda eff, rec: eff
    exs = _examples(model, records)
    for ex, rec in zip(exs, records):
        eff0 = list(rec.get("effect_labels", [0] * len(
            rec["policy_input"]["candidates"])))
        ex["effect_labels"] = np.asarray(labels_fn(eff0, rec), dtype=np.int32)
    rng = random.Random(seed)
    mx.random.seed(seed)
    from ucm.v1.transfer import FinetuneConfig as _FC
    _fc = _FC(updates=updates, seed=seed)
    opt = AdamW(learning_rate=_fc.lr, weight_decay=_fc.weight_decay)
    idx = list(range(len(exs)))
    t0 = time.time()
    curve = []
    skipped = 0
    for step in range(1, updates + 1):
        rng.shuffle(idx)
        chunk = [exs[i] for i in idx[:min(32, len(exs))]] or exs
        batch = _collate_p2(chunk)
        import numpy as _np
        K = batch["cand_mask"].shape[1]
        eff = _np.zeros((len(chunk), K), dtype=_np.int32)
        for i, e in enumerate(chunk):
            li = len(e["effect_labels"])
            eff[i, :li] = e["effect_labels"]
        eff_labels = mx.array(eff)

        def loss_fn():
            logits, eff_logits = model(batch)
            lp = set_bc_loss(logits, batch["labels"], batch["cand_mask"])
            if eff_logits is None:
                return lp
            # CE manuel (mlx.core n'expose pas cross_entropy): logits (B,K,C)
            logp = eff_logits - mx.stop_gradient(
                mx.max(eff_logits, axis=-1, keepdims=True))
            logp = logp - mx.log(mx.sum(mx.exp(logp), axis=-1, keepdims=True))
            C = eff_logits.shape[-1]
            tgt = mx.array(__import__("numpy").eye(C, dtype="float32")[
                __import__("numpy").asarray(eff_labels.tolist())])
            le = -mx.mean(mx.sum(tgt * logp, axis=-1))
            return lp + model.effect_lambda * le

        loss, grads = nn.value_and_grad(model, loss_fn)()
        from ucm.v1.runner import _grad_global_norm
        from mlx.utils import tree_map as _tm
        import math as _math
        lv, gn = float(loss), float(_grad_global_norm(grads))
        if not (_math.isfinite(lv) and _math.isfinite(gn)):
            # garde divergence (20:23): batch non-fini ⇒ update SAUTÉ,
            # compté, publié — jamais de paramètres empoisonnés
            skipped += 1
            curve.append({"step": step, "loss": None, "skipped": True})
            # ABORT PRÉCOCE (lead 20:36-2): >50% de batches non-finis dans
            # les 20 premiers updates ⇒ divergence STRUCTURELLE — abort
            # avec diagnostic, jamais 391/400 silencieux
            if step <= 20 and skipped > step // 2:
                raise RuntimeError(
                    f"DIVERGENCE PRÉCOCE cellule: {skipped}/{step} batches "
                    f"non-finis dans les {step} premiers updates — abort "
                    f"avant empoisonnement (bug structurel, pas du bruit)")
            continue
        scale = mx.minimum(1.0, _fc.clip_norm / mx.maximum(gn, 1e-12))
        grads = _tm(lambda g: g * scale, grads)
        if not model.head_enabled:
            # HEAD-NEVER-TRAINED: l'optimiseur ne voit QUE la base (subtree) —
            # AdamW wd ne peut pas dériver une tête désactivée (leçon no-op)
            sub = tree_map(lambda g: g, dict(grads)["base"])
            opt.update(model.base, sub)
        else:
            opt.update(model, tree_map(lambda g: g, grads))
        mx.eval(loss, model.parameters())
        if step == 1 or step % 100 == 0:
            curve.append({"step": step, "loss": float(loss)})
    fin = [c for c in curve if c.get("loss") is not None]
    return {"updates": updates, "wall_s": round(time.time() - t0, 1),
            "loss_first": fin[0]["loss"] if fin else None,
            "loss_last": fin[-1]["loss"] if fin else None,
            "loss_curve": curve, "n_records": len(records),
            "n_skipped_batches": skipped}


def build_arm_records(ds_eps, family_key, arm, seed):
    """Records du bras: D0 = bruts; D1-mixed = 30% épisodes avec contexte;
    D1-shuffled = 30% avec cibles permutées."""
    from ucm.p2.records import episode_records
    from ucm.p2 import context_arms as ca
    from ucm.p2.generator import split_family
    eps = ds_eps[family_key]
    sp = split_family(eps)
    train = sp["train"]
    ids = [e["episode_id"] for e in train]
    d1_ids = ca.d1_episode_ids(ids) if arm != "D0" else set()
    rng = __import__("random").Random(ca.CTX_SEED + seed)
    records = []
    for ep in train:
        ctx_ents = ctx_rels = None
        if ep["episode_id"] in d1_ids:
            # historique: effets observés d'AUTRES épisodes (in-context)
            others = [e for e in train if e["episode_id"] != ep["episode_id"]]
            src = others[rng.randrange(len(others))] if others else ep
            from ucm.dsl.core import DSLInterpreter
            from ucm.dsl.tgk_program import TGK_DSL_PROGRAM
            interp = DSLInterpreter(TGK_DSL_PROGRAM, src["layout_tables"])
            effs = ca.extract_effects(interp, None, src["task"])
            if arm == "D1-mixed":
                ctx_ents, ctx_rels = ca.build_context_entities(effs, rng=rng)
            else:
                ctx_ents, ctx_rels = ca.build_context_entities_shuffled(effs, rng=rng)
        records.extend(episode_records(ep, ctx_ents, ctx_rels))
    return records, sp


class P2Policy:
    """Policy closed-loop P2 (tensorize_obs_p2, head ignoré à la décision)."""

    def __init__(self, model, name, seed):
        self.model, self.name, self.rng = model, name, __import__("random").Random(seed)

    def __call__(self, obs):
        import numpy as np
        import mlx.core as mx
        from ucm.p2.tensorize_p2 import tensorize_obs_p2
        from ucm.model.tensorize import randomize_obs_order
        from ucm.model.gnn_b import NEG_INF
        obs2 = randomize_obs_order(obs, self.rng)
        ex = tensorize_obs_p2(obs2)
        ex["labels"] = None
        batch = _collate_p2([ex])
        logits, _ = self.model(batch)
        out = np.asarray(logits[0].tolist())
        order = np.argsort(-out)
        best = int(order[0])
        return obs2["candidates"][best]


def eval_cell(model, eval_eps, arm, seed, budget_journal):
    """Closed-loop sur les tâches TEST, contexte du bras attaché aux obs."""
    import random as _r
    from ucm.env.tinygraph import TinyGraphKey
    from ucm.eval.rollout import run_episode
    from ucm.p2.records import _native_layout
    from ucm.p2 import context_arms as ca
    rng = _r.Random(EVAL_SEED + seed)
    raws = []
    for ep in eval_eps:
        lay = _native_layout(ep["layout_tables"])
        env = TinyGraphKey(lay)
        base_obs = env.reset(ep["task"])
        # contexte du bras (MEME construction que l'entraînement)
        if arm != "D0":
            others = [e for e in eval_eps if e["episode_id"] != ep["episode_id"]]
            src = others[rng.randrange(len(others))] if others else ep
            from ucm.dsl.core import DSLInterpreter
            from ucm.dsl.tgk_program import TGK_DSL_PROGRAM
            interp = DSLInterpreter(TGK_DSL_PROGRAM, src["layout_tables"])
            effs = ca.extract_effects(interp, None, src["task"])
            crng = _r.Random(ca.CTX_SEED + seed)
            if arm == "D1-mixed":
                ce, cr = ca.build_context_entities(effs, rng=crng)
            else:
                ce, cr = ca.build_context_entities_shuffled(effs, rng=crng)
            obs0 = ca.augment_obs(base_obs, ce, cr)
        else:
            obs0 = base_obs
        pol = _AugPolicy(model, obs0, arm, seed)
        r = run_episode(env, pol, ep["episode_id"], EVAL_SEED + seed,
                        f"p2-{arm}", d_star=ep["d_star"])
        raws.append({"cell_arm": arm, "episode_id": ep["episode_id"],
                     "success": bool(r.success), "outcome": r.outcome,
                     "length": r.length, "n_invalid": r.n_invalid,
                     "d_star": ep["d_star"], "L_star": ep["d_star"] + 1,
                     "family": "-".join(ep["family"])})
        budget_journal.append({"ts": time.time(), "cell_eval": ep["episode_id"]})
    succ = sum(1 for r in raws if r["success"])
    return raws, succ / max(len(raws), 1)


class _AugPolicy:
    """Attache le contexte figé à CHAQUE observation de l'épisode."""

    def __init__(self, model, base_obs_ctx, arm, seed):
        self.inner = P2Policy(model, f"p2-{arm}", EVAL_SEED + seed)
        self.ctx_e = [e for e in base_obs_ctx["entities"]
                      if e["type"].startswith("ctx_")]
        self.ctx_r = [r for r in base_obs_ctx["relations"]
                      if r["pred"] == ca_rel()]

    def __call__(self, obs):
        obs2 = dict(obs)
        obs2["entities"] = list(obs["entities"]) + list(self.ctx_e)
        obs2["relations"] = list(obs["relations"]) + list(self.ctx_r)
        return self.inner(obs2)


def ca_rel():
    from ucm.p2.context_arms import CTX_REL
    return CTX_REL


def run_p2(out_dir=_OUT, ts=None, updates=UPDATES, run=True,
           families=("DROP-AT-d2-12", "DROP-HAVE-d2-12"),
           seeds=SEEDS, arms=ARMS):
    if not run:
        return {"status": "plan", "cells": len(families) * len(arms) * len(seeds),
                "arms": list(arms), "seeds": list(seeds), "updates": updates}
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    os.makedirs(out_dir, exist_ok=True)
    ts = ts or time.strftime("%Y%m%dT%H%M%S")
    ckpt_dir = os.environ.get("UCM_P2_CKPT_DIR") or os.path.join(out_dir, "ckpts-run4")
    os.makedirs(ckpt_dir, exist_ok=True)
    ds = load_dataset()
    budget_journal = []

    # GATE AVANT TOUT ENTRAÎNEMENT (protocole §3 — jamais de run sous gate échoué)
    from ucm.p2.gate import gate_equal_info
    gate = gate_equal_info(ds["episodes"])

    results = {"ts": ts, "updates": updates, "gate": gate, "cells": {}}
    for fam in families:
        for arm in arms:
            for seed in seeds:
                from ucm.p2.model_p2 import P2Model
                mx.random.seed(seed)
                model = P2Model(d=144)
                records, sp = build_arm_records(ds["episodes"], fam, arm, seed)
                train_meta = train_cell(model, records, updates, seed)
                budget_journal.append({"ts": time.time(), "cell": f"{fam}|{arm}|s{seed}",
                                       "wall_s": train_meta["wall_s"]})
                raws, succ = eval_cell(model, sp["test"], arm, seed, budget_journal)
                cell = {"family": fam, "arm": arm, "seed": seed,
                        "success_rate": succ, "raws": raws, "train": train_meta}
                # ckpt O_EXCL
                import mlx.nn as _nn
                import mlx.core as _mx
                ck = os.path.join(ckpt_dir, f"{fam}-{arm}-s{seed}.npz")
                if os.path.exists(ck):
                    raise RuntimeError(f"ckpt O_EXCL: {ck}")
                _mx.savez(ck, **dict(_nn.utils.tree_flatten(model.parameters())))
                results["cells"][f"{fam}|{arm}|s{seed}"] = cell
                print(f"[p2] {fam}|{arm}|s{seed}: {succ:.2f} "
                      f"({train_meta['wall_s']}s)")

    # persistance O_EXCL: run + budget journal
    rpath = os.path.join(out_dir, f"p2-run-{ts}.json")
    _fd = os.open(rpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(results, fh, sort_keys=True, indent=1, default=str)
    jpath = os.path.join(out_dir, f"p2-budget-{ts}.jsonl")
    _fd = os.open(jpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        for e in budget_journal:
            fh.write(json.dumps(e, sort_keys=True) + "\n")
    results["persisted"] = rpath
    results["budget_journal"] = jpath
    return results


if __name__ == "__main__":
    import sys
    r = run_p2(run="--plan" not in sys.argv)
    print(json.dumps({k: v for k, v in r.items() if k != "cells"},
                     indent=1, default=str)[:1200])


# ---------------------------------------------------------------------------
# Q4 helpers (lead 05:59) — évals facteurs
# ---------------------------------------------------------------------------

def _eval_eps_gate5(sp):
    """Épisodes test d'une famille P2 au format gate5 (ref, hash, d0, recs)."""
    from ucm.p2.records import _native_layout
    from ucm.env.tinygraph import TinyGraphKey
    out = []
    for ep in sp["test"]:
        lay = _native_layout(ep["layout_tables"])
        env = TinyGraphKey(lay)
        obs = env.reset(ep["task"])
        out.append((ep["episode_id"], ep["layout_hash"], ep["d_star"],
                    [{"policy_input": obs,
                      "provenance": {"episode_ref": ep["episode_id"],
                                     "layout_hash": ep["layout_hash"],
                                     "step": 0}}]))
    return out


def _evaluate_tgk_native_records(model, sp):
    from ucm.eval.factorial_2x2 import _evaluate_tgk
    res = _evaluate_tgk(model, _eval_eps_gate5(sp), seed=8000, arm="q4")
    return sum(1 for r in res if r.success) / len(res)


def _evaluate_p2_simple(model, sp):
    import numpy as np
    from ucm.p2.tensorize_p2 import tensorize_obs_p2
    from ucm.p2.records import _native_layout
    from ucm.env.tinygraph import TinyGraphKey
    from ucm.eval.rollout import run_episode

    class Simple:
        def __init__(self, m): self.m = m
        def __call__(self, obs):
            ex = tensorize_obs_p2(obs); ex["labels"] = None
            lg, _ = self.m(_collate_p2([ex]))
            return obs["candidates"][int(np.argmax(np.asarray(lg[0].tolist())))]

    succ = 0
    for ep in sp["test"]:
        lay = _native_layout(ep["layout_tables"])
        env = TinyGraphKey(lay)
        env.reset(ep["task"])
        r = run_episode(env, Simple(model), ep["episode_id"], 1, "q4",
                        d_star=ep["d_star"])
        succ += r.success
    return succ / len(sp["test"])


def _evaluate_ckpt_without_context(ckpt_path, sp):
    """Charge un ckpt run4 et évalue SANS contexte (obs D0 pures)."""
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_default_device(mx.cpu)
    from ucm.p2.model_p2 import P2Model
    m = P2Model(d=144)
    w = dict(mx.load(ckpt_path))
    own = dict(nn.utils.tree_flatten(m.parameters()))
    if set(own) != set(w):
        raise RuntimeError(f"ckpt keys mismatch: {ckpt_path}")
    m.update(nn.utils.tree_unflatten(sorted(w.items())))
    mx.eval(m.parameters())
    return _evaluate_p2_simple(m, sp)
