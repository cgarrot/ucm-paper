#!/usr/bin/env python3
"""Generate the paper figures as SVG (no external data beyond curated constants).

Usage: python tools/make_figures.py
Outputs: paper/figures/fig1-stack.svg ... fig6-results.svg
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper", "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#1a1a2e"
MUTED = "#6b7280"
BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
LIGHT = "#f3f4f6"
GREY = "#9ca3af"
TEAL = "#0d9488"
PURPLE = "#7c3aed"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "svg.fonttype": "none",
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
})


def box(ax, x, y, w, h, text, fc=LIGHT, ec=INK, fs=9, weight="normal", tc=INK, radius=0.02, lw=1.0, ha="center"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={radius}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha=ha, va="center", fontsize=fs, weight=weight,
            color=tc, zorder=3, linespacing=1.35)
    return b


def arrow(ax, p1, p2, color=INK, lw=1.2, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}", zorder=1, linestyle=ls,
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)


def blank_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    svg_path = os.path.join(OUT, name)
    png_path = os.path.join(OUT, name.replace(".svg", ".png"))
    fig.savefig(svg_path, format="svg", bbox_inches="tight", transparent=False, facecolor="white")
    fig.savefig(png_path, format="png", dpi=160, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print("wrote", svg_path, "+ png")


# ---------------------------------------------------------------- fig 1: stack
def fig1():
    fig, ax = blank_axes((7.2, 5.4))
    box(ax, 3.0, 8.6, 4.0, 0.9, "VOICE\n(external transcripts)", fc="#eef2ff", ec=BLUE)
    box(ax, 3.0, 7.0, 4.0, 1.1, "BIG MODEL\nUNDERSTAND + DECOMPOSE\n(structured goal: genres)", fc="#eef2ff", ec=BLUE)
    box(ax, 2.2, 4.6, 5.6, 1.7,
        "UCM — EXECUTE\n~0.7 M params · ~1 ms/decision\nnon-autoregressive typed decisions\nabstains when blocked",
        fc="#ecfdf5", ec=GREEN, weight="bold", fs=9, lw=1.2)
    box(ax, 3.0, 2.2, 4.0, 1.0, "MEASURED RESULT\nsuccess · cost · latency · abstention", fc="#fefce8", ec=AMBER)
    box(ax, 0.2, 5.25, 1.5, 1.3, "structured\nworld\n(entities,\nrelations,\ngoal,\ncandidates)", fc="white", ec=MUTED, fs=7.5)
    box(ax, 8.3, 5.25, 1.5, 1.3, "actions\nMOVE / CLICK\nTYPE / …\nSTOP", fc="white", ec=MUTED, fs=7.5)
    arrow(ax, (5, 8.6), (5, 8.1))
    arrow(ax, (5, 7.0), (5, 6.3))
    arrow(ax, (5, 4.6), (5, 3.2))
    arrow(ax, (2.2, 5.75), (1.7, 5.75), color=MUTED)
    arrow(ax, (7.8, 5.75), (8.3, 5.75), color=MUTED)
    arrow(ax, (4.2, 2.2), (2.0, 4.95), color=RED, rad=0.25, ls="--")
    ax.text(0.9, 3.4, "on failure:\ntargeted\nre-decomposition", color=RED, fontsize=8, ha="center")
    ax.text(5.0, 0.7, "UCM is the execution layer — the big model never sits in the fast path",
            ha="center", fontsize=9, style="italic", color=MUTED)
    save(fig, "fig1-stack.svg")


# ---------------------------------------------------------------- fig 2: worlds
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # --- TGK
    ax = axes[0]
    ax.set_title("TinyGraphKey — relational world", fontsize=11, weight="bold", color=INK)
    rooms = {"v0": (1.5, 7.6), "v1": (4.6, 8.3), "v2": (7.6, 7.4), "v3": (3.0, 4.6), "v4": (7.0, 4.2), "v5": (5.0, 2.2)}
    edges = [("v0", "v1", False), ("v1", "v2", True), ("v0", "v3", False), ("v1", "v3", False),
             ("v3", "v4", False), ("v2", "v4", False), ("v3", "v5", False), ("v4", "v5", False)]
    for a, b, door in edges:
        x1, y1 = rooms[a]; x2, y2 = rooms[b]
        col = RED if door else GREY
        lw = 2.2 if door else 1.0
        ax.plot([x1, x2], [y1, y2], color=col, lw=lw, zorder=1)
        if door:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.plot([mx], [my], marker="s", ms=9, mec=RED, mfc="white", mew=1.6, zorder=3)
            ax.text(mx + 0.25, my + 0.25, "door (locked)", color=RED, fontsize=8)
    for name, (x, y) in rooms.items():
        ax.add_patch(Circle((x, y), 0.52, fc="#eef2ff", ec=BLUE, lw=1.2, zorder=2))
        ax.text(x, y, name, ha="center", va="center", fontsize=8, color=INK, zorder=3)
    # agent + objects
    ax.plot([1.5], [7.6], marker="^", ms=9, color=GREEN, zorder=4)
    ax.text(1.5, 8.4, "agent", color=GREEN, fontsize=8, ha="center")
    ax.plot([3.0], [4.6 + 0.75], marker="o", ms=7, color=AMBER, zorder=4)
    ax.text(3.0, 5.6, "key", color=AMBER, fontsize=8, ha="center")
    ax.plot([7.0], [4.2 + 0.75], marker="s", ms=7, color=PURPLE, zorder=4)
    ax.text(7.0, 5.2, "parcel", color=PURPLE, fontsize=8, ha="center")
    ax.text(5, 0.6, "goals: REACH(room) · HAVE(obj) · AT(obj, room)\n"
                    "actions: MOVE PICK DROP UNLOCK STOP · K = R + 6 · invalid = paid",
            ha="center", fontsize=8, color=MUTED)

    # --- SIW
    ax = axes[1]
    ax.set_title("SIW — software-like world", fontsize=11, weight="bold", color=INK)
    box(ax, 0.8, 7.2, 2.6, 1.6, "view: Home\nnav_edge → Form", fc="#eef2ff", ec=BLUE, fs=8)
    box(ax, 4.2, 6.6, 5.0, 3.0, "", fc="white", ec=INK)
    ax.text(6.7, 9.30, "view: Form page", fontsize=8, weight="bold", color=INK, ha="center")
    box(ax, 4.5, 8.0, 2.3, 0.7, "field: email\n(filled?)", fc=LIGHT, ec=MUTED, fs=7)
    box(ax, 7.0, 8.0, 1.9, 0.7, "field: name\n(filled?)", fc=LIGHT, ec=MUTED, fs=7)
    box(ax, 4.5, 7.1, 2.3, 0.7, "select: x0\n3 options", fc=LIGHT, ec=MUTED, fs=7)
    box(ax, 7.0, 7.1, 1.9, 0.7, "button\n“annuler”\n(onclick: submit)", fc="#fef2f2", ec=RED, fs=6.5)
    box(ax, 2.4, 3.4, 5.2, 2.2,
        "dialog (open?)\n· confirm/dismiss ≡ close\n· labels are random:\na submit button\nmay read “cancel”",
        fc="#fffbeb", ec=AMBER, fs=8)
    box(ax, 0.4, 0.6, 9.2, 1.9,
        "goals: VIEW · SET · CHOOSE · SUBMITTED\n"
        "actions: NAVIGATE CLICK TYPE SELECT STOP\n"
        "K = V+B+F+O+1 (30–60)\n"
        "invalid ≠ valid no-op · select is irreversible",
        fc=LIGHT, ec=MUTED, fs=8)
    arrow(ax, (3.4, 8.0), (4.2, 8.0), color=BLUE)
    ax.text(3.75, 8.62, "NAVIGATE", fontsize=6.2, color=BLUE, ha="center")
    save(fig, "fig2-worlds.svg")


# ---------------------------------------------------------------- fig 3: DSL
def fig3():
    fig, ax = blank_axes((7.6, 4.6))
    box(ax, 0.4, 7.6, 3.4, 1.6, "native engines\ncode/ucm/env/tinygraph.py\ncode/ucm/env/siw.py", fc="#eef2ff", ec=BLUE, fs=8.5)
    box(ax, 6.2, 7.6, 3.4, 1.6, "DSL programs (data)\ntgk_program.py · siw_program.py", fc="#ecfdf5", ec=GREEN, fs=8.5)
    box(ax, 3.4, 4.6, 3.2, 1.7, "generic interpreter\nreset-only · guards\neffect groups\none successor fn", fc=LIGHT, ec=INK, fs=8.5)
    box(ax, 1.2, 2.0, 3.4, 1.5, "differential harness\ncandidates · validity\nsuccessor · goal\nd* · optimal set (action-wise)", fc="#f5f3ff", ec=PURPLE, fs=8)
    box(ax, 5.6, 2.0, 3.2, 1.5, "mutation suite\n3 injected bugs\n≥1 divergence each", fc="#fef2f2", ec=RED, fs=8)
    box(ax, 2.6, 0.1, 5.0, 0.9, "≥ 10 000 states/world · equality proven · mutations detected", fc="#ecfdf5", ec=GREEN, fs=9, weight="bold")
    arrow(ax, (3.8, 7.6), (4.6, 6.3), color=BLUE)
    arrow(ax, (6.2, 7.6), (5.4, 6.3), color=GREEN)
    arrow(ax, (4.2, 4.6), (3.0, 3.5), color=PURPLE)
    arrow(ax, (5.8, 4.6), (7.0, 3.5), color=RED)
    arrow(ax, (3.0, 2.0), (4.4, 1.0), color=INK)
    arrow(ax, (7.0, 2.0), (5.6, 1.0), color=INK)
    ax.text(5.0, 5.45, "the interpreter knows no world", ha="center", fontsize=8.5, style="italic", color=MUTED)
    save(fig, "fig3-dsl.svg")


# ---------------------------------------------------------------- fig 4: architecture
def fig4():
    fig, ax = blank_axes((9.8, 5.0))
    # input contract
    box(ax, 0.2, 7.0, 2.5, 2.4,
        "policy_input\nentities (typed)\nrelations (typed)\ngoal predicate + refs\ncandidates (goal-blind)",
        fc="#eef2ff", ec=BLUE, fs=8)
    box(ax, 0.2, 4.0, 2.5, 2.3,
        "tensor contract\nnodes [N,D_IN]\nedges (2 directed)\ngoal [pred, 2 refs]\ncandidates [K,2] + mask",
        fc=LIGHT, ec=INK, fs=8)
    box(ax, 0.2, 1.2, 2.5, 2.0,
        "supervision / provenance\nSEPARATE channels\nanti-leak: cannot change tensors",
        fc="#fefce8", ec=AMBER, fs=8)
    # models
    box(ax, 3.4, 6.6, 2.6, 1.4, "Deep Sets A\n1-hop pooling\n366 337 params\nGATE-1 FAIL (ties)", fc="#fef2f2", ec=RED, fs=8)
    box(ax, 3.4, 4.4, 2.6, 1.6, "GNN B144\n3 message rounds\nresidual + LayerNorm\nd = 144\n694 513 params", fc="#ecfdf5", ec=GREEN, fs=8, weight="bold")
    box(ax, 3.4, 2.3, 2.6, 1.5, "heads (shared)\nmasked mean pool → ctx\ngoal_mlp\nscore_mlp (shared)", fc=LIGHT, ec=INK, fs=8)
    # output
    box(ax, 7.0, 5.6, 2.7, 2.0, "one parallel pass\nlogit per candidate\nsoftmax over candidates\nset-valued BC loss\n−log Σ_{a∈A*} p(a)", fc="#f5f3ff", ec=PURPLE, fs=8)
    box(ax, 7.0, 3.4, 2.7, 1.5, "greedy decode\nargmax → typed action\nSTOP is an action", fc="#fefce8", ec=AMBER, fs=8)
    box(ax, 7.0, 1.4, 2.7, 1.4, "no autoregression\nno text tokens\nno embedded labels/ids", fc=LIGHT, ec=MUTED, fs=8)
    arrow(ax, (2.7, 7.9), (3.4, 7.4), color=BLUE)
    arrow(ax, (2.7, 6.6), (3.4, 5.3), color=BLUE)
    arrow(ax, (2.7, 4.6), (3.4, 3.2), color=INK)
    arrow(ax, (6.0, 7.2), (7.0, 6.7), color=RED)
    arrow(ax, (6.0, 5.2), (7.0, 6.2), color=GREEN)
    arrow(ax, (6.0, 3.2), (7.0, 3.9), color=INK)
    arrow(ax, (8.35, 5.6), (8.35, 4.9), color=PURPLE)
    arrow(ax, (8.35, 3.4), (8.35, 2.8), color=AMBER)
    save(fig, "fig4-architecture.svg")


# ---------------------------------------------------------------- fig 5: gates
def fig5():
    fig, ax = blank_axes((10.0, 4.2))
    def chip(x, y, label, sub, color):
        box(ax, x, y, 1.62, 1.5, f"{label}\n", fc="white", ec=color, lw=1.6)
        ax.text(x + 0.81, y + 0.95, label, ha="center", va="center", fontsize=9, weight="bold", color=color)
        ax.text(x + 0.81, y + 0.45, sub, ha="center", va="center", fontsize=7, color=INK, linespacing=1.3)

    y1 = 7.4
    chip(0.15, y1, "GATE-0", "canon sealed\n3 machines\nbyte-identical", GREEN)
    chip(2.01, y1, "GATE-1", "A FAILS\n0.9658 ties\nat bit level", RED)
    chip(3.87, y1, "GATE-5", "B144 elected\n+62.2 pp\ncost 1.921×", GREEN)
    chip(5.73, y1, "GATE-2", "97.50 %\nCI [96.5;98.5]\n3 160 eps", GREEN)
    chip(7.59, y1, "GATE-3", "99.35 %\nreserved cell\n494 eps", GREEN)
    for x in (1.77, 3.63, 5.49, 7.35):
        arrow(ax, (x, y1 + 0.75), (x + 0.24, y1 + 0.75), color=MUTED)

    y2 = 4.4
    chip(0.15, y2, "M3", "controls 0.0\nperm 200/200\nheuristic 82.6 %", GREEN)
    chip(2.01, y2, "V1", "EXPLORATORY\nreclassified\n(instrument)", AMBER)
    chip(3.87, y2, "V1-bis", "NOT DEMONSTRATED\nsaturated ≤+0.73 pp\nSTOP fixed 0/72k", RED)
    chip(5.73, y2, "S2b", "KILL\n−7.7 vs +15.8\nretention PASS", RED)
    chip(7.59, y2, "P2", "INDETERMINATE\n+34 pp not repl.\ncontext: not testable", RED)

    y3 = 1.4
    chip(0.15, y3, "S5", "product line\n44/44 · 40/40\nstub Jev", GREEN)
    chip(2.01, y3, "Probe", "web compiler\n5.86 %\nswitch rule", RED)
    box(ax, 4.4, y3, 5.4, 1.5, "three open locks\n(1) strict composition  (2) depth via short search  (3) real-software compiler",
        fc="#f5f3ff", ec=PURPLE, fs=9, weight="bold")
    ax.text(5.0, 0.45, "green = established · amber = exploratory · red = not established / killed",
            ha="center", fontsize=8, color=MUTED, style="italic")
    save(fig, "fig5-gates.svg")


# ---------------------------------------------------------------- fig 6: results
def fig6():
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    dpi_note = "curated artifacts; see results/json"

    # (a) success by d*
    ax = axes[0, 0]
    d = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    s = [1.00, 1.00, 0.985, 0.927, 0.938, 0.957, 1.00, 0.846, 0.625, 0.40, 0.25, 0.463, 0.545, 0.40, 0.00, 0.091]
    n = [69, 56, 68, 41, 32, 23, 7, 13, 8, 5, 4, 41, 22, 15, 8, 11]
    bars = ax.bar(d, s, color=[BLUE if x <= 12 else RED for x in d], alpha=0.85, width=0.72)
    for xi, si, ni in zip(d, s, n):
        ax.text(xi, si + 0.03, f"n{ni}", ha="center", fontsize=6.5, color=MUTED)
    ax.axvspan(12.5, 17.5, color=RED, alpha=0.06)
    ax.axvline(12.5, color=RED, ls="--", lw=0.9)
    ax.text(15.0, 0.62, "G4 band", color=RED, fontsize=8, ha="center")
    ax.set_ylim(0, 1.12); ax.set_xlabel("optimal distance d*"); ax.set_ylabel("closed-loop success")
    ax.set_title("(a) The depth wall (G3 blue / G4 red)", fontsize=10, weight="bold")
    ax.set_xticks(d); ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    # (b) capacity/cost frontier
    ax = axes[0, 1]
    models = [("A\n366k", 366337, 0.259, 1.00, GREEN, "ok"),
              ("B144\n695k", 694513, 0.881, 1.92, GREEN, "elected"),
              ("B160\n856k", 856161, 0.912, 2.04, RED, "cost fail"),
              ("B192\n1.23M", 1230145, 0.895, 2.31, RED, "cost fail")]
    for name, p, succ, cost, col, note in models:
        ax.scatter(p / 1000, succ * 100, s=220, color=col, zorder=3, edgecolor="white", lw=1.2)
        off = 14 if name.startswith("A") else -34
        ax.annotate(f"{name}\n{cost:.2f}×", (p / 1000, succ * 100), textcoords="offset points",
                    xytext=(0, off), ha="center", fontsize=7.5, color=col, weight="bold")
    ax.axhline(100, color=MUTED, lw=0.5, ls=":")
    ax.set_xlabel("parameters (thousands)"); ax.set_ylabel("val depth-stratum success (%)")
    ax.set_title("(b) Capacity / cost frontier", fontsize=10, weight="bold")
    ax.set_ylim(18, 104); ax.grid(alpha=0.25, lw=0.5)
    ax.text(1050, 30, "labels: cost ratio\n(gate ≤ 2×)", fontsize=7, color=MUTED)

    # (c) V1-bis arms
    ax = axes[1, 0]
    ks = [64, 128, 256]
    arms = {"scratch": ([97.10, 98.18, 99.18], GREY), "pretrained": ([97.90, 98.08, 99.10], BLUE),
            "control-validity": ([96.22, 96.98, 98.95], AMBER), "control-null": ([96.37, 98.17, 99.34], TEAL)}
    for name, (vals, col) in arms.items():
        ax.plot(ks, vals, marker="o", color=col, lw=1.6, ms=5, label=name)
    ax.set_xticks(ks); ax.set_xticklabels(["64", "128", "256"])
    ax.set_xlabel("adaptation budget k (episodes)"); ax.set_ylabel("success (%)")
    ax.set_title("(c) V1-bis transfer: saturated regime", fontsize=10, weight="bold")
    ax.set_ylim(95.5, 100); ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.25, lw=0.5)
    ax.text(150, 95.72, "saturation margins: 2.90 / 1.82 / 0.82 pp\n"
                        "detectable ≤ +0.73 pp at k=256",
            fontsize=6.8, color=RED, ha="center")

    # (d) P2 ablation
    ax = axes[1, 1]
    labels = ["A\nimitation", "B\neffects\ncorrect", "C\neffects\nshuffled", "D\nsimple\ntarget"]
    vals = [35.4, 41.0, 29.9, 52.8]
    cols = [GREY, BLUE, RED, GREEN]
    ax.bar(labels, vals, color=cols, alpha=0.88, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, weight="bold")
    ax.set_ylim(0, 68); ax.set_ylabel("closed-loop success (%)")
    ax.set_title("(d) Effect-head ablation (12 seeds)", fontsize=10, weight="bold")
    ax.text(0.5, 57, "B−A = +5.6 pp (CI ∋ 0)   B−D = −11.8 pp\nverdict INDETERMINATE",
            ha="center", fontsize=7.5, color=RED)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    fig.suptitle("UCM results overview — statuses, instruments and negatives", fontsize=12, weight="bold", y=0.995)
    fig.text(0.99, 0.005, dpi_note, ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    save(fig, "fig6-results.svg")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("done")
