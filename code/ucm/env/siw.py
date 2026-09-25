"""SIW — Monde d'Interaction Synthétique (V1, spec §12.6, docs/SIW-SPEC.md).

Monde logiciel-like déterministe pleinement observable: vues navigables, widgets
typés (boutons/champs/selects/formulaires/dialogues/distracteurs), labels
arbitraires sans lien sémantique avec le rôle, actions argumentées, ordre de
validation, deux classes d'improductivité (invalide payante vs no-op valide).

Contrat d'information V0 §5.1 inchangé: observe()/candidates() ne renvoient QUE
policy_input. Le succès est attribué par l'évaluateur (goal_satisfied), jamais
par le modèle. Chaque décision (STOP inclus) coûte un pas.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Optional

ACTION_TYPES = ("NAVIGATE", "CLICK", "TYPE", "SELECT", "STOP")
GOAL_PREDICATES = ("VIEW", "SET", "CHOOSE", "SUBMITTED")
HORIZON_DEFAULT = 64

# Vocabulaire fermé de labels (~60 mots, paires d'alias) — AUCUN lien sémantique
# avec les rôles: un bouton submit peut porter "annuler".
_LABEL_WORDS = [
    "accueil", "retour", "suivant", "precedent", "envoyer", "annuler",
    "valider", "refuser", "ouvrir", "fermer", "enregistrer", "effacer",
    "ajouter", "retirer", "chercher", "filtrer", "trier", "modifier",
    "consulter", "detail", "liste", "tableau", "profil", "compte",
    "message", "notification", "alerte", "confirmation", "avertissement",
    "chargement", "parametre", "option", "preference", "affichage",
    "edition", "creation", "suppression", "duplication", "partage",
    "export", "import", "impression", "archive", "corbeille", "favori",
    "etiquette", "dossier", "fichier", "lien", "piece", "document",
    "formulaire", "champ", "case", "menu", "volet", "onglet", "barre",
    "bouton", "zone", "section", "en-tete", "pied",
]
_LABEL_ALIASES = {  # alias → mot canonique (les deux tirables)
    "envoie": "envoyer", "valide": "valider", "annule": "annuler",
    "sauver": "enregistrer", "suppr": "suppression", "prefs": "preference",
    "params": "parametre", "accueil2": "accueil",
}

WIDGET_TYPES = ("button", "field", "select", "option", "form", "dialog")


class SIWLayoutError(ValueError):
    pass


class SIWLayout:
    """Structure statique: vues, widgets, relations, machine à états déclarée.

    Les labels sont tirés du vocabulaire fermé SANS lien avec les rôles.
    L'isomorphisme (WS-B) porte sur widgets+FSM, JAMAIS sur les labels.
    """

    def __init__(self, spec: dict):
        """spec = {
          views: [view_id...],
          nav_edges: [[v, v]...],           # bidirectionnels implicites
          widgets: [
            {id, type, view, kind?, form?, select?, options? [opt_id...],
             submit_for?: form_id, in_dialog?: dialog_id}
          ...],
          labels: {entity_id: label}
        }"""
        self.views = list(spec["views"])
        self.nav_edges = [tuple(e) for e in spec["nav_edges"]]
        self.widgets = {w["id"]: dict(w) for w in spec["widgets"]}
        self.labels = dict(spec.get("labels", {}))

        self._check()

        self._adj: dict[str, set[str]] = {v: set() for v in self.views}
        for a, b in self.nav_edges:
            self._adj[a].add(b)
            self._adj[b].add(a)

    def _check(self):
        if not (2 <= len(self.views) <= 5):
            raise SIWLayoutError("2..5 views (spec §4)")
        vs = set(self.views)
        for a, b in self.nav_edges:
            if a not in vs or b not in vs or a == b:
                raise SIWLayoutError("bad nav edge")
        if len({frozenset(e) for e in self.nav_edges}) != len(self.nav_edges):
            raise SIWLayoutError("duplicate nav edge")
        if not self._connected():
            raise SIWLayoutError("views graph must be connected")
        for w in self.widgets.values():
            if w["type"] not in WIDGET_TYPES:
                raise SIWLayoutError(f"bad type {w['type']}")
            if (w["type"] not in ("option",) and not w.get("in_dialog")
                    and w.get("view") not in vs):
                raise SIWLayoutError(f"widget {w['id']} not on a view")
            if w["type"] == "button" and w.get("kind") not in (
                    "submit", "confirm", "dismiss", "none"):
                raise SIWLayoutError("bad button kind")
        # options: exactement un select parent, pas de view propre
        for w in self.widgets.values():
            if w["type"] == "option":
                if w.get("select") not in self.widgets or \
                        self.widgets[w["select"]]["type"] != "select":
                    raise SIWLayoutError("option without select parent")
            if w.get("in_dialog") is not None:
                d = self.widgets.get(w["in_dialog"])
                if d is None or d["type"] != "dialog":
                    raise SIWLayoutError(f"in_dialog ghost: {w['in_dialog']}")
            if w.get("submit_for") is not None:
                f = self.widgets.get(w["submit_for"])
                if f is None or f["type"] != "form":
                    raise SIWLayoutError(f"submit_for ghost: {w['submit_for']}")
            if w.get("form") is not None:
                f = self.widgets.get(w["form"])
                if f is None or f["type"] != "form":
                    raise SIWLayoutError(f"form ref ghost: {w.get('form')}")
        for w in self.widgets.values():
            if w["type"] == "select":
                opts = [o for o in self.widgets.values()
                        if o.get("select") == w["id"]]
                if not (2 <= len(opts) <= 4):
                    raise SIWLayoutError("select needs 2..4 options")
            if w["type"] == "form":
                members = [x for x in self.widgets.values()
                           if x.get("form") == w["id"]]
                n_f = sum(1 for x in members if x["type"] == "field")
                n_s = sum(1 for x in members if x["type"] == "select")
                if not (2 <= n_f <= 4) or n_s > 1:
                    raise SIWLayoutError("form: 2..4 fields, <=1 select (spec §4)")
                subs = [b for b in self.widgets.values()
                        if b.get("submit_for") == w["id"]]
                if len(subs) != 1:
                    raise SIWLayoutError("form needs exactly one submit button")

    def _connected(self) -> bool:
        seen = {self.views[0]}
        stack = [self.views[0]]
        while stack:
            cur = stack.pop()
            for a, b in self.nav_edges:
                if a == cur and b not in seen:
                    seen.add(b); stack.append(b)
                elif b == cur and a not in seen:
                    seen.add(a); stack.append(a)
        return len(seen) == len(self.views)

    def adjacent(self, a: str, b: str) -> bool:
        return b in self._adj[a]

    def widgets_on(self, view: str) -> list[dict]:
        return [w for w in self.widgets.values() if w.get("view") == view]

    def form_members(self, form_id: str) -> list[dict]:
        return [w for w in self.widgets.values() if w.get("form") == form_id]

    def canonical(self) -> str:
        """Forme canonique SANS labels (isomorphisme: widgets+FSM, §2.3)."""
        payload = {
            "views": sorted(self.views),
            "nav": sorted(sorted(e) for e in self.nav_edges),
            "widgets": sorted(
                (w["id"], w["type"], w.get("view"), w.get("kind"),
                 w.get("form"), w.get("select"), w.get("submit_for"),
                 w.get("in_dialog"),
                 tuple(sorted(w.get("options", []))))
                for w in self.widgets.values()),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def layout_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# État dynamique
# --------------------------------------------------------------------------- #

class SIWState:
    __slots__ = ("view", "filled", "chosen", "dialog_open", "submitted")

    def __init__(self, view: str, filled: frozenset, chosen: dict,
                 dialog_open: bool, submitted: frozenset):
        self.view = view
        self.filled = filled            # ids des fields remplis
        self.chosen = chosen            # select_id -> option_id
        self.dialog_open = dialog_open
        self.submitted = submitted      # ids des forms submitted

    def key(self):
        return (self.view, frozenset(self.filled), tuple(sorted(self.chosen.items())),
                self.dialog_open, frozenset(self.submitted))

    def __eq__(self, o):
        return isinstance(o, SIWState) and self.key() == o.key()

    def __hash__(self):
        return hash(self.key())

    def copy(self):
        return SIWState(self.view, self.filled, dict(self.chosen),
                        self.dialog_open, self.submitted)


def goal_satisfied(state: SIWState, layout: SIWLayout, goal: dict) -> bool:
    pred, args = goal["predicate"], goal.get("args", {})
    if pred == "VIEW":
        return state.view == args["view"]
    if pred == "SET":
        return args["field"] in state.filled
    if pred == "CHOOSE":
        return state.chosen.get(args["select"]) == args["option"]
    if pred == "SUBMITTED":
        return args["form"] in state.submitted
    raise ValueError(f"unknown predicate {pred}")


def task_goal(pred: str, **args) -> dict:
    if pred not in GOAL_PREDICATES:
        raise ValueError(f"unknown predicate {pred}")
    return {"predicate": pred, "args": args}


# --------------------------------------------------------------------------- #
# Environnement
# --------------------------------------------------------------------------- #

class SIW:
    """MDP déterministe. Invalides = no-op payants avec flag invalid;
    no-ops VALIDES (distracteurs, submit prématuré) = coût sans effet."""

    def __init__(self, layout: SIWLayout, horizon: int = HORIZON_DEFAULT):
        self.layout = layout
        self.horizon = horizon
        self.goal: Optional[dict] = None
        self.state: Optional[SIWState] = None
        self.step_count = 0
        self.terminal = False
        self.last_result: Optional[str] = None

    # -- cycle de vie --------------------------------------------------------- #

    def reset(self, task: dict) -> dict:
        init, goal = task["init"], task["goal"]
        self._validate_goal(goal)
        self.goal = goal
        self.state = SIWState(
            view=init["view"],
            filled=frozenset(init.get("filled", [])),
            chosen=dict(init.get("chosen", {})),
            dialog_open=bool(init.get("dialog_open", False)),
            submitted=frozenset(init.get("submitted", [])),
        )
        self._check_state(init)
        if self.state.dialog_open and not any(
                w["type"] == "dialog" for w in self.layout.widgets.values()):
            raise SIWLayoutError("dialog_open=True sans dialog dans le layout")
        self.step_count = 0
        self.terminal = False
        self.last_result = None
        return self.observe()

    def _validate_goal(self, goal: dict):
        if not isinstance(goal, dict) or set(goal) != {"predicate", "args"}:
            raise SIWLayoutError("goal: clés exactes ['predicate','args']")
        pred, args = goal["predicate"], goal["args"]
        expected = {"VIEW": {"view"}, "SET": {"field"},
                    "CHOOSE": {"select", "option"},
                    "SUBMITTED": {"form"}}.get(pred)
        if expected is None:
            raise SIWLayoutError(f"unknown predicate {pred!r}")
        if not isinstance(args, dict) or set(args) != expected:
            raise SIWLayoutError(f"goal args: {sorted(expected)} requis")
        W = self.layout.widgets
        V = self.layout.views
        for key, valid in (("view", V), ("field", W), ("select", W),
                           ("option", W), ("form", W)):
            if key in args and args[key] not in valid:
                raise SIWLayoutError(f"unknown ref {args[key]!r}")
        # m2 (fix régression tagi-5: "view" vit dans V, pas dans W)
        for key, treq in (("field", "field"), ("select", "select"),
                          ("option", "option"), ("form", "form")):
            if key in args and W[args[key]]["type"] != treq:
                raise SIWLayoutError(
                    f"goal ref {key}={args[key]!r} doit être de type {treq}")
        if "view" in args and args["view"] not in V:
            raise SIWLayoutError(f"goal view inconnue {args['view']!r}")
        if pred == "CHOOSE" and W[args["option"]].get("select") != args["select"]:
            raise SIWLayoutError("option n'appartient pas au select du but")

    def _check_state(self, init: dict):
        s, lay = self.state, self.layout
        if s.view not in lay.views:
            raise SIWLayoutError("bad init view")
        for f in s.filled:
            if lay.widgets.get(f, {}).get("type") != "field":
                raise SIWLayoutError("bad filled ref")
        for sel, opt in s.chosen.items():
            w = lay.widgets.get(sel, {})
            if w.get("type") != "select" or lay.widgets.get(opt, {}).get("select") != sel:
                raise SIWLayoutError("bad chosen pair")
        for fm in s.submitted:
            if lay.widgets.get(fm, {}).get("type") != "form":
                raise SIWLayoutError("bad submitted ref")

    # -- observation (policy_input UNIQUEMENT) -------------------------------- #

    def observe(self) -> dict:
        s, lay = self.state, self.layout
        entities = [{"id": v, "type": "view", "attrs": {}} for v in lay.views]
        for w in lay.widgets.values():
            attrs = {}
            if w["type"] == "button":
                attrs["onclick_kind"] = w.get("kind", "none")
            elif w["type"] == "field":
                attrs["filled"] = w["id"] in s.filled
            elif w["type"] == "select":
                attrs["chosen_option_ref"] = s.chosen.get(w["id"])
            elif w["type"] == "form":
                attrs["status"] = ("submitted" if w["id"] in s.submitted
                                   else "complete" if self._form_complete(w["id"])
                                   else "draft")
            elif w["type"] == "dialog":
                attrs["open"] = s.dialog_open
            if w["id"] in lay.labels:
                attrs["label"] = lay.labels[w["id"]]  # arbitraire, §2.3
            ent = {"id": w["id"], "type": w["type"], "attrs": attrs}
            if w["type"] != "option":
                ent["view"] = w.get("view")  # référence structurelle observable
            entities.append(ent)

        relations = []
        for a, b in lay.nav_edges:
            relations.append({"subj": a, "pred": "nav_edge", "obj": b})
            relations.append({"subj": b, "pred": "nav_edge", "obj": a})
        for w in lay.widgets.values():
            if w["type"] != "option" and w.get("view"):
                relations.append({"subj": w["id"], "pred": "on_view",
                                  "obj": w["view"]})
            if w.get("form"):
                relations.append({"subj": w["id"], "pred": "part_of",
                                  "obj": w["form"]})
            if w.get("select"):
                relations.append({"subj": w["id"], "pred": "option_of",
                                  "obj": w["select"]})
            if w.get("submit_for"):
                relations.append({"subj": w["id"], "pred": "submits",
                                  "obj": w["submit_for"]})
            if w.get("in_dialog"):
                relations.append({"subj": w["id"], "pred": "in_dialog",
                                  "obj": w["in_dialog"]})
        relations.append({"subj": "agent", "pred": "current_view",
                          "obj": s.view})
        for f in s.filled:
            relations.append({"subj": f, "pred": "filled", "obj": "agent"})
        for sel, opt in s.chosen.items():
            relations.append({"subj": opt, "pred": "chosen", "obj": sel})

        return {
            "schema_version": "0.2",
            "entities": entities,
            "relations": relations,
            "goal": {"predicate": self.goal["predicate"],
                     "args": dict(self.goal.get("args", {}))},
            "candidates": self.candidates(),
        }

    def candidates(self) -> list[dict]:
        """Énumération syntaxique goal-blind/state-blind partielle (§3.1):
        NAVIGATE×views, CLICK×boutons, TYPE×fields, SELECT×options, STOP.
        K = V + B + F + O + 1."""
        lay = self.layout
        out = [{"action": "NAVIGATE", "arg": v} for v in lay.views]
        out += [{"action": "CLICK", "arg": b["id"]} for b in
                lay.widgets.values() if b["type"] == "button"]
        out += [{"action": "TYPE", "arg": f["id"]} for f in
                lay.widgets.values() if f["type"] == "field"]
        out += [{"action": "SELECT", "arg": o["id"]} for o in
                lay.widgets.values() if o["type"] == "option"]
        out += [{"action": "STOP", "arg": None}]
        return out

    # -- dynamique ------------------------------------------------------------ #

    def _visible(self, widget_id: str) -> bool:
        w = self.layout.widgets[widget_id]
        if w["type"] == "option":
            return self._visible(w["select"])
        if w.get("in_dialog"):
            return self.state.dialog_open
        return w.get("view") == self.state.view

    def _form_complete(self, form_id: str) -> bool:
        s = self.state
        for m in self.layout.form_members(form_id):
            if m["type"] == "field" and m["id"] not in s.filled:
                return False
            if m["type"] == "select" and m["id"] not in s.chosen:
                return False
        return True

    def _is_valid(self, action: dict) -> tuple[bool, str]:
        """(valid, effect) — effect ∈ {'none','navigate','click','type',
        'select','stop'} ; valid=False ⇒ invalide payant ; valid=True +
        effect='none' ⇒ no-op VALIDE (distracteur, submit prématuré...)."""
        s, lay = self.state, self.layout
        kind, arg = action["action"], action["arg"]
        if kind == "NAVIGATE":
            if arg == s.view or not lay.adjacent(s.view, arg):
                return False, "none"
            return True, "navigate"
        if kind == "CLICK":
            w = lay.widgets.get(arg)
            if w is None or w["type"] != "button" or not self._visible(arg):
                return False, "none"
            if w.get("kind") == "submit":
                form = w.get("submit_for")
                return True, ("click" if self._form_complete(form)
                              and form not in s.submitted else "none")
            if w.get("kind") == "confirm":
                return True, ("click" if s.dialog_open else "none")
            if w.get("kind") == "dismiss":
                return True, ("click" if s.dialog_open else "none")
            return True, "none"  # distracteur: valide, sans effet
        if kind == "TYPE":
            w = lay.widgets.get(arg)
            if (w is None or w["type"] != "field" or not self._visible(arg)
                    or arg in s.filled):
                return False, "none"
            # §5 (révisé): TYPE unaire, remplissage binaire, validité
            # goal-blind — la règle initiale 'uniquement le field du but'
            # rendait SUBMITTED insolvable (attrapé par smoke test).
            return True, "type"
        if kind == "SELECT":
            w = lay.widgets.get(arg)
            if w is None or w["type"] != "option" or not self._visible(arg):
                return False, "none"
            if w.get("select") in s.chosen:
                return False, "none"
            return True, "select"
        if kind == "STOP":
            return True, "stop"
        return False, "none"

    def execute(self, action: dict) -> dict:
        if self.terminal:
            raise RuntimeError("episode is terminal")
        if action not in self.candidates():
            raise ValueError(f"action not in candidates: {action}")
        valid, effect = self._is_valid(action)
        kind, arg = action["action"], action["arg"]
        s = self.state

        if kind == "STOP":
            self.step_count += 1
            self.terminal = True
            self.last_result = ("success" if goal_satisfied(s, self.layout, self.goal)
                                else "premature_stop")
            return {"valid": True, "terminal": True, "result": self.last_result,
                    "step": self.step_count, "effect": "stop"}

        if effect == "navigate":
            s.view = arg
        elif effect == "type":
            s.filled = s.filled | {arg}
        elif effect == "select":
            w = self.layout.widgets[arg]
            s.chosen = dict(s.chosen); s.chosen[w["select"]] = arg
        elif effect == "click":
            w = self.layout.widgets[arg]
            if w.get("kind") == "submit":
                s.submitted = s.submitted | {w["submit_for"]}
                s.dialog_open = False
            elif w.get("kind") in ("confirm", "dismiss"):
                s.dialog_open = False
        # effect 'none': rien (invalide OU no-op valide)

        self.step_count += 1
        if self.step_count >= self.horizon:
            self.terminal = True
            self.last_result = "timeout"
            return {"valid": valid, "terminal": True, "result": "timeout",
                    "step": self.step_count, "effect": effect}
        return {"valid": valid, "terminal": False,
                "result": "ok" if valid else "invalid", "step": self.step_count,
                "effect": effect}

    # -- côté évaluateur UNIQUEMENT ------------------------------------------- #

    def goal_satisfied(self) -> bool:
        return goal_satisfied(self.state, self.layout, self.goal)

    def state_hash(self) -> str:
        payload = json.dumps(
            {"layout": self.layout.layout_hash(),
             "state": [self.state.view, sorted(self.state.filled),
                       sorted(self.state.chosen.items()),
                       self.state.dialog_open, sorted(self.state.submitted)]},
            sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Générateur de layouts (seedé, §4 de la spéc)
# --------------------------------------------------------------------------- #

def generate_siw_layout(rng: random.Random, n_views: int,
                        with_dialog: Optional[bool] = None,
                        k_bounds: tuple[int, int] = (30, 60),
                        profile: str = "standard") -> SIWLayout:
    """Générateur déterministe: arbre de vues + extra arêtes, formulaires,
    distracteurs majoritaires, labels tirés du vocabulaire fermé.

    profile="small" (re-scoping §4.5 après inventaire): 1 form, 2 fields,
    2 options — espace état-but ~1000 couples/layout pour que les budgets
    de transfert k=100..10000 couvrent une fraction interprétable; K reste
    garanti 30-60 par remplissage de distracteurs (qui n'inflent PAS
    l'espace d'états — difficulté many-candidates conservée)."""
    views = [f"vw{i}" for i in range(n_views)]
    rng.shuffle(views)
    nav = []
    for i in range(1, n_views):
        nav.append((views[i], views[rng.randrange(i)]))
    seen = {frozenset(e) for e in nav}
    extra = rng.randint(0, max(0, n_views // 2))
    added = tries = 0
    while added < extra and tries < 20 * n_views:
        tries += 1
        a, b = rng.sample(views, 2)
        if frozenset((a, b)) not in seen:
            nav.append((a, b)); seen.add(frozenset((a, b))); added += 1

    widgets: list[dict] = []
    if profile == "small":
        n_forms = 1
        n_fields_range = (2, 2)
        n_opts_range = (2, 2)
    else:
        n_forms = rng.randint(1, 2)
        n_fields_range = (2, 4)
        n_opts_range = (2, 4)
    form_views = rng.sample(views, min(n_forms, len(views)))
    for fi in range(n_forms):
        fid = f"fm{fi}"
        fview = form_views[fi]
        widgets.append({"id": fid, "type": "form", "view": fview})
        n_fields = rng.randint(*n_fields_range)
        for j in range(n_fields):
            widgets.append({"id": f"fd{fi}_{j}", "type": "field",
                            "view": fview, "form": fid})
        n_opts = rng.randint(*n_opts_range)
        sid = f"sl{fi}"
        widgets.append({"id": sid, "type": "select", "view": fview,
                        "form": fid})
        for j in range(n_opts):
            widgets.append({"id": f"op{fi}_{j}", "type": "option",
                            "select": sid})
        widgets.append({"id": f"bt_s{fi}", "type": "button", "view": fview,
                        "kind": "submit", "submit_for": fid})
        # NB: pas d'attr form sur le bouton submit (submits_for suffit)

    use_dialog = (rng.random() < 0.30) if with_dialog is None else with_dialog
    if use_dialog:
        did = "dlg0"
        widgets.append({"id": did, "type": "dialog", "view": rng.choice(views)})
        widgets.append({"id": "bt_c", "type": "button", "kind": "confirm",
                        "in_dialog": did})
        widgets.append({"id": "bt_d", "type": "button", "kind": "dismiss",
                        "in_dialog": did})
    # distracteurs: combler jusqu'à la borne inférieure K (§4), plafonnés
    def _K():
        return (len(views)
                + sum(1 for w in widgets if w["type"] == "button")
                + sum(1 for w in widgets if w["type"] == "field")
                + sum(1 for w in widgets if w["type"] == "option") + 1)
    target_k = rng.randint(k_bounds[0], k_bounds[1])
    j = 0
    while _K() < target_k:
        widgets.append({"id": f"bt_x{j}", "type": "button",
                        "view": rng.choice(views), "kind": "none"})
        j += 1

    # labels: vocabulaire fermé, alias inclus, SANS lien avec les rôles
    pool = _LABEL_WORDS + list(_LABEL_ALIASES)
    labels = {}
    for w in widgets:
        if w["type"] in ("button", "field", "select", "form", "dialog"):
            labels[w["id"]] = rng.choice(pool)

    spec = {"views": views, "nav_edges": [list(e) for e in nav],
            "widgets": widgets, "labels": labels}
    lay = SIWLayout(spec)
    K = _K()
    if not (k_bounds[0] <= K <= k_bounds[1] + 2):
        raise SIWLayoutError(f"K={K} hors bornes {k_bounds}")
    return lay


def sample_task(rng: random.Random, layout: SIWLayout) -> tuple[SIWState, dict]:
    """(état initial, but) aléatoires — l'appelant vérifie reachability/bande.
    dialog_open seulement si le layout POSSÈDE un dialog (cohérence d'état)."""
    lay = layout
    init_view = rng.choice(lay.views)
    has_dialog = any(w["type"] == "dialog" for w in lay.widgets.values())
    goal = _sample_goal(rng, layout)
    return SIWState(init_view, frozenset(), {},
                    has_dialog and rng.random() < 0.50, frozenset()), goal


def _sample_goal(rng: random.Random, layout: SIWLayout) -> dict:
    lay = layout
    pred = rng.choice(GOAL_PREDICATES)
    if pred == "VIEW":
        return task_goal("VIEW", view=rng.choice(lay.views))
    if pred == "SET":
        fields = [w for w in lay.widgets.values() if w["type"] == "field"]
        return task_goal("SET", field=rng.choice(fields)["id"])
    if pred == "CHOOSE":
        sels = [w for w in lay.widgets.values() if w["type"] == "select"]
        sel = rng.choice(sels)
        opts = [o for o in lay.widgets.values() if o.get("select") == sel["id"]]
        return task_goal("CHOOSE", select=sel["id"],
                         option=rng.choice(opts)["id"])
    forms = [w for w in lay.widgets.values() if w["type"] == "form"]
    return task_goal("SUBMITTED", form=rng.choice(forms)["id"])
