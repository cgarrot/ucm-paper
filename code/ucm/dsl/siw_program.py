"""Programme DSL: SIW — le natif devient la SECONDE INSTANCE.

Sémantique verbatim de ucm/env/siw.py:
  NAVIGATE v: garde = v≠view ∧ nav(view,v); effet set view
  CLICK b (button): garde = visible(b); effet SI kind=submit ∧ form_complete
      ∧ form∉submitted → s_add submitted, dialog=False;
      SI kind∈{confirm,dismiss} ∧ dialog_open → dialog=False; sinon no-op valide
  TYPE f: garde = visible(f) ∧ f∉filled; effet s_add filled
  SELECT o: garde = visible(o) ∧ select(o) ∉ chosen-keys; effet d_set chosen
  STOP: valide; succès = but
  Visibilité: option → visible(select parent); in_dialog → dialog_open;
      sinon w.view == state.view
  Buts: VIEW v | SET f | CHOOSE sel=opt | SUBMITTED form
Layout tables: views, nav (paires), buttons/fields/options (listes),
w_type/w_kind/w_view/w_select/w_submit_for/w_in_dialog (dicts),
form_fields/form_selects (dicts — membres par formulaire).
"""
from __future__ import annotations

SIW_DSL_PROGRAM = {
    "schema": "ucm-dsl/1.0",
    "world": "siw",
    "state": ["view", "filled", "chosen", "dialog_open", "submitted"],
    "reset": {
        "view": ["task", "init", "view"],
        "filled": ["if", ["is_none", ["task", "init", "filled"]],
                   ["const", []], ["task", "init", "filled"]],
        "chosen": ["if", ["is_none", ["task", "init", "chosen"]],
                   ["const", {}], ["task", "init", "chosen"]],
        "dialog_open": ["if", ["is_none", ["task", "init", "dialog_open"]],
                        ["const", False], ["task", "init", "dialog_open"]],
        "submitted": ["if", ["is_none", ["task", "init", "submitted"]],
                      ["const", []], ["task", "init", "submitted"]],
    },
    "state_check": {
        "view_known": ["mem1", "views", ["field", "view"]],
    },
    "actions": [
        {"name": "NAVIGATE",
         "arg_source": ["each", "views"],
         "guard": [["not", ["eq", ["field", "view"], ["param", "arg"]]],
                   ["mem2", "nav", ["field", "view"], ["param", "arg"]]],
         "effects": [["set", "view", ["param", "arg"]]]},
        {"name": "CLICK",
         "arg_source": ["each", "buttons"],
         "guard": [
             # bouton visible: in_dialog → dialog_open, sinon sur la vue courante
             ["or",
              ["and",
               ["not", ["is_none", ["tget", "w_in_dialog", ["param", "arg"]]]],
               ["field", "dialog_open"]],
              ["and",
               ["is_none", ["tget", "w_in_dialog", ["param", "arg"]]],
               ["eq", ["tget", "w_view", ["param", "arg"]], ["field", "view"]]]]],
         # effet UNIQUEMENT pour submit complet / confirm-dismiss avec dialog;
         # sinon no-op VALIDE (distracteur, submit prématuré - semantique 3.1)
         "when": ["or",
                  ["and", ["eq", ["tget", "w_kind", ["param", "arg"]], ["const", "submit"]],
                   ["s_all_in", "filled", ["tget", "form_fields",
                                           ["tget", "w_submit_for", ["param", "arg"]]]],
                   ["d_all_has", "chosen",
                    ["tget", "form_selects", ["tget", "w_submit_for", ["param", "arg"]]]],
                   ["not", ["smem", "submitted", ["tget", "w_submit_for", ["param", "arg"]]]]],
                  ["and",
                   ["or", ["eq", ["tget", "w_kind", ["param", "arg"]], ["const", "confirm"]],
                    ["eq", ["tget", "w_kind", ["param", "arg"]], ["const", "dismiss"]]],
                   ["field", "dialog_open"]]],
         # groupes PAR KIND (natif: submit -> s_add submitted + dialog=False;
         # confirm/dismiss -> dialog=False SEULEMENT)
         "effect_groups": [
             {"when": ["eq", ["tget", "w_kind", ["param", "arg"]], ["const", "submit"]],
              "do": [["s_add", "submitted", ["tget", "w_submit_for", ["param", "arg"]]],
                     ["set", "dialog_open", ["const", False]]]},
             {"when": ["or", ["eq", ["tget", "w_kind", ["param", "arg"]], ["const", "confirm"]],
                       ["eq", ["tget", "w_kind", ["param", "arg"]], ["const", "dismiss"]]],
              "do": [["set", "dialog_open", ["const", False]]]}]},
        {"name": "TYPE",
         "arg_source": ["each", "fields"],
         "guard": [
             # champ visible: in_dialog -> dialog_open, sinon sur la vue courante
             ["or",
              ["and",
               ["not", ["is_none", ["tget", "w_in_dialog", ["param", "arg"]]]],
               ["field", "dialog_open"]],
              ["and",
               ["is_none", ["tget", "w_in_dialog", ["param", "arg"]]],
               ["eq", ["tget", "w_view", ["param", "arg"]], ["field", "view"]]]],
             ["not", ["smem", "filled", ["param", "arg"]]]],
         "effects": [["s_add", "filled", ["param", "arg"]]]},
        {"name": "SELECT",
         "arg_source": ["each", "options"],
         "guard": [
             # option visible via son select parent: in_dialog -> dialog_open,
             # sinon select sur la vue courante
             ["or",
              ["and",
               ["not", ["is_none", ["tget", "w_in_dialog",
                                    ["tget", "w_select", ["param", "arg"]]]]],
               ["field", "dialog_open"]],
              ["and",
               ["is_none", ["tget", "w_in_dialog",
                            ["tget", "w_select", ["param", "arg"]]]],
               ["eq", ["tget", "w_view", ["tget", "w_select", ["param", "arg"]]],
                ["field", "view"]]]],
             ["not", ["d_has", "chosen", ["tget", "w_select", ["param", "arg"]]]]],
         "effects": [["d_set", "chosen", ["tget", "w_select", ["param", "arg"]],
                      ["param", "arg"]]]},
    ],
    "goal": ["or",
             ["and", ["eq", ["task", "goal", "predicate"], ["const", "VIEW"]],
              ["eq", ["field", "view"], ["task", "goal", "args", "view"]]],
             ["and", ["eq", ["task", "goal", "predicate"], ["const", "SET"]],
              ["smem", "filled", ["task", "goal", "args", "field"]]],
             ["and", ["eq", ["task", "goal", "predicate"], ["const", "CHOOSE"]],
              ["eq", ["dget", "chosen", ["task", "goal", "args", "select"]],
               ["task", "goal", "args", "option"]]],
             ["and", ["eq", ["task", "goal", "predicate"], ["const", "SUBMITTED"]],
              ["smem", "submitted", ["task", "goal", "args", "form"]]]],
}


def layout_tables_from_siw(layout) -> dict:
    """Tables DSL depuis un SIWLayout natif."""
    w = layout.widgets
    buttons = [x["id"] for x in w.values() if x["type"] == "button"]
    fields = [x["id"] for x in w.values() if x["type"] == "field"]
    options = [x["id"] for x in w.values() if x["type"] == "option"]
    nav = sorted({tuple(sorted((a, b))) for a, b in layout.nav_edges})
    form_fields, form_selects = {}, {}
    for x in w.values():
        if x["type"] == "form":
            mem = layout.form_members(x["id"])
            form_fields[x["id"]] = [m["id"] for m in mem if m["type"] == "field"]
            form_selects[x["id"]] = [m["id"] for m in mem if m["type"] == "select"]
    return {
        "views": list(layout.views),
        "nav": [list(p) for p in nav],
        "buttons": buttons, "fields": fields, "options": options,
        "w_type": {k: v["type"] for k, v in w.items()},
        "w_kind": {k: v.get("kind") for k, v in w.items()},
        "w_view": {k: v.get("view") for k, v in w.items()},
        "w_select": {k: v.get("select") for k, v in w.items()},
        "w_submit_for": {k: v.get("submit_for") for k, v in w.items()},
        "w_in_dialog": {k: v.get("in_dialog") for k, v in w.items()},
        "form_fields": form_fields, "form_selects": form_selects,
    }
