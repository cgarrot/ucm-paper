"""DSL RÉEL (P2-A, lead 17:54-2) — interpréteur GÉNÉRIQUE, jalon 1.

Le natif devient UNE INSTANCE du DSL, pas la référence. L'interpréteur ne
connaît NI les pièces, NI les clés, NI SIW: il ne sait évaluer que le
langage d'expressions et exécuter les effets déclarés par le PROGRAMME.

Construction par RESET UNIQUEMENT (tue la classe B1 — aucun état injecté
sans reset): DSLState n'est constructible que via interpreter.reset(task).

Langage (arbres JSON, zéro eval Python):
  expr := ["const", v] | ["field", f] | ["param", p] | ["task", k1, k2, …]
        | ["layout", k]                      # table du layout (rooms/adj/…)
        | ["and", e…] | ["or", e…] | ["not", e]
        | ["eq", e1, e2] | ["is_none", e]
        | ["if", c, a, b]
        | ["mem1", table, e]                 # e ∈ layout[table] (liste plate)
        | ["mem2", table, e1, e2]            # paire (ordre-libre) ∈ table
        | ["smem", field, e]                 # e ∈ state[field] (ensemble=liste)
effet := ["set", field, expr]
       | ["s_add", field, expr] | ["s_rm", field, expr]

Extensible à l'état de croyance: les champs sont opaques pour l'interpréteur
(un champ peut être une distribution — les opérations set/mem restent valides).
"""
from __future__ import annotations

import json

DSL_SCHEMA = "ucm-dsl/1.0"


class DSLError(ValueError):
    pass


class DSLState:
    """État IMMABLE (canonical: champs triés, ensembles = listes triées).
    Constructible UNIQUEMENT par l'interpréteur (reset) — pas d'injection."""

    __slots__ = ("fields",)

    def __init__(self, fields: dict):
        object.__setattr__(self, "fields",
                           {k: _canon(v) for k, v in fields.items()})

    def key(self) -> tuple:
        return tuple(sorted((k, _freeze(v)) for k, v in self.fields.items()))

    def __repr__(self):
        return f"DSLState({self.fields})"


def _canon(v):
    if isinstance(v, list):
        return sorted(v, key=lambda x: json.dumps(x, sort_keys=True, default=str))
    return v


def _freeze(v) -> str:
    return json.dumps(v, sort_keys=True, default=str)


class DSLInterpreter:
    """Interpréte UN programme. reset(task) est la SEULE porte d'entrée."""

    def __init__(self, program: dict, layout: dict):
        if program.get("schema") != DSL_SCHEMA:
            raise DSLError(f"schema {program.get('schema')!r} != {DSL_SCHEMA}")
        self.program = program
        self.layout = layout
        self.state: DSLState | None = None
        self.task = None
        self.terminal = False
        self.step_count = 0

    # -- reset: SEULE construction d'état (B1) -------------------------------- #

    def reset(self, task: dict) -> dict:
        self.task = task
        fields = {}
        for fname, expr in self.program["reset"].items():
            fields[fname] = self._eval(expr)
        self.state = DSLState(fields)
        self._check()
        self.terminal = False
        self.step_count = 0
        return self.observe()

    def _check(self):
        st = self.program.get("state_check", {})
        for fname, expr in st.items():
            if not self._eval(expr):
                raise DSLError(f"state check failed: {fname}")

    # -- langage -------------------------------------------------------------- #

    def _eval(self, e):
        op = e[0]
        if op == "const":
            return e[1]
        if op == "field":
            name = e[1] if isinstance(e[1], str) else self._eval(e[1])
            return self.state.fields[name]
        if op == "param":
            return self.params[e[1]]
        if op == "task":
            v = self.task
            for k in e[1:]:
                v = v[k]
            return v
        if op == "layout":
            return self.layout[e[1]]
        if op == "and":
            return all(self._eval(x) for x in e[1:])
        if op == "or":
            return any(self._eval(x) for x in e[1:])
        if op == "not":
            return not self._eval(e[1])
        if op == "eq":
            return self._eval(e[1]) == self._eval(e[2])
        if op == "is_none":
            return self._eval(e[1]) is None
        if op == "if":
            return self._eval(e[2]) if self._eval(e[1]) else self._eval(e[3])
        if op == "mem1":
            return self._eval(e[2]) in self.layout[e[1]]
        if op == "mem2":
            a, b = self._eval(e[2]), self._eval(e[3])
            return [a, b] in self.layout[e[1]] or [b, a] in self.layout[e[1]]
        if op == "smem":
            return self._eval(e[2]) in self.state.fields[e[1]]
        if op == "tget":                     # table dict: tget[table, key]
            return self.layout[e[1]].get(self._eval(e[2]))
        if op == "dget":                     # dict field: dget[field, key]
            return self.state.fields[e[1]].get(self._eval(e[2]))
        if op == "d_has":                    # clé présente dans dict field
            return self._eval(e[2]) in self.state.fields[e[1]]
        if op == "s_all_in":                 # ∀x ∈ liste: x ∈ field (ensemble)
            return all(x in self.state.fields[e[1]] for x in self._eval(e[2]))
        if op == "d_all_has":                # ∀k ∈ liste: k ∈ clés(dict field)
            return all(k in self.state.fields[e[1]] for k in self._eval(e[2]))
        raise DSLError(f"op inconnue: {op}")

    # -- candidats (énumération syntaxique, ordre canonique) ------------------ #

    def candidates(self) -> list[dict]:
        out = []
        for act in self.program["actions"]:
            for arg in self._enum_args(act):
                out.append({"action": act["name"], "arg": arg})
        out.append({"action": "STOP", "arg": None})
        return out

    def _enum_args(self, act: dict) -> list:
        src = act.get("arg_source")
        if src is None:
            return [None]
        if src[0] == "const_list":
            return list(src[1])
        table = self.layout[src[1]]
        if src[0] == "each":
            return list(table)
        if src[0] == "each_pair":   # éléments d'un tableau de paires
            return [x for pair in table for x in pair]
        raise DSLError(f"arg_source inconnu: {src}")

    # -- dynamique ------------------------------------------------------------ #

    def _guard_ok(self, act: dict, cand: dict) -> bool:
        self.params = {"arg": cand["arg"]}
        if cand["action"] == "STOP":
            return True
        return all(self._eval(g) for g in act.get("guard", []))

    def _apply(self, act: dict, cand: dict):
        """Applique les effets si `when` (condition d'effet) est vraie —
        sinon no-op VALIDE (sémantique UI: activé mais sans effet)."""
        self.params = {"arg": cand["arg"]}
        when = act.get("when")
        if when is not None and not self._eval(when):
            return DSLState(dict(self.state.fields))
        fields = dict(self.state.fields)
        groups = act.get("effect_groups", act.get("effects", []))
        if groups and isinstance(groups[0], dict):
            pass  # groupes conditionnels
        else:
            groups = [{"do": groups}]  # effets plats → un groupe unique
        for group in groups:
            # groupe: {"when": expr|null, "do": [effets]} — conditions PAR groupe
            gwhen = group.get("when")
            gdo = group["do"]
            if gwhen is not None and not self._eval(gwhen):
                continue
            for eff in gdo:
                if eff[0] == "set":
                    fname = eff[1] if isinstance(eff[1], str) else self._eval(eff[1])
                    fields[fname] = _canon(self._eval(eff[2]))
                elif eff[0] == "s_add":
                    s = set(fields[eff[1]])
                    s.add(self._eval(eff[2]))
                    fields[eff[1]] = sorted(s)
                elif eff[0] == "s_rm":
                    s = set(fields[eff[1]])
                    s.discard(self._eval(eff[2]))
                    fields[eff[1]] = sorted(s)
                elif eff[0] == "d_set":       # dict field: d_set[field, key, val]
                    d = dict(fields[eff[1]])
                    d[self._eval(eff[2])] = self._eval(eff[3])
                    fields[eff[1]] = d
                else:
                    raise DSLError(f"effet inconnu: {eff[0]}")
        return DSLState(fields)

    def goal_satisfied(self) -> bool:
        return self._eval(self.program["goal"])

    def _action_spec(self, name: str) -> dict:
        for a in self.program["actions"]:
            if a["name"] == name:
                return a
        raise DSLError(f"action inconnue: {name}")

    def successor(self, cand: dict):
        """(valid, next_state | None) — pur, sans toucher self.state.
        Utilisé par le BFS ET par execute (une seule implémentation)."""
        spec = self._action_spec(cand["action"]) if cand["action"] != "STOP" else None
        saved, saved_t = self.state, self.terminal
        try:
            valid = self._guard_ok(spec, cand) if spec else True
            if not valid or spec is None:  # STOP ou invalide → pas de successeur
                return valid, None
            return True, self._apply(spec, cand)
        finally:
            self.state, self.terminal = saved, saved_t

    def execute(self, action: dict) -> dict:
        if self.terminal:
            raise RuntimeError("episode is terminal")
        if action not in self.candidates():
            raise DSLError(f"action hors candidats: {action}")
        spec = (self._action_spec(action["action"])
                if action["action"] != "STOP" else None)
        valid = self._guard_ok(spec, action) if spec else True
        if action["action"] == "STOP":
            self.step_count += 1
            self.terminal = True
            res = "success" if self.goal_satisfied() else "premature_stop"
            return {"valid": True, "terminal": True, "result": res,
                    "step": self.step_count}
        if valid:
            self.state = self._apply(spec, action)
        self.step_count += 1
        terminal = self.step_count >= self.program.get("horizon", 64)
        result = ("ok" if valid else "invalid") if not terminal else "timeout"
        return {"valid": valid, "terminal": terminal, "result": result,
                "step": self.step_count}

    # -- observation ---------------------------------------------------------- #

    def observe(self) -> dict:
        return self.program["observe"](self) if callable(
            self.program.get("observe")) else {
            "schema_version": "dsl/1.0",
            "state_view": {k: v for k, v in self.state.fields.items()},
            "goal": self.task["goal"],
            "candidates": self.candidates(),
        }
