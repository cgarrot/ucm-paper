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
    fig, ax = blank_axes((10.0, 4.4))
    def chip(x, y, label, sub, color):
        box(ax, x, y, 1.62, 1.75, "", fc="white", ec=color, lw=1.6)
        ax.text(x + 0.81, y + 1.32, label, ha="center", va="center", fontsize=9, weight="bold", color=color)
        ax.text(x + 0.81, y + 0.58, sub, ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.35)

    y1 = 7.2
    chip(0.15, y1, "GATE-0", "canon sealed\n3 machines identical", GREEN)
    chip(2.01, y1, "GATE-1", "A FAILS 0.9658\nbit-exact ties", RED)
    chip(3.87, y1, "GATE-5", "B144 elected\n+62.2 pp", GREEN)
    chip(5.73, y1, "GATE-2", "97.50 %\n3 160 episodes", GREEN)
    chip(7.59, y1, "GATE-3", "99.35 %\nreserved cell", GREEN)
    for x in (1.77, 3.63, 5.49, 7.35):
        arrow(ax, (x, y1 + 0.87), (x + 0.24, y1 + 0.87), color=MUTED)

    y2 = 4.1
    chip(0.15, y2, "M3", "controls 0.0\nheuristic 82.6 %", GREEN)
    chip(2.01, y2, "V1", "EXPLORATORY\nreclassified", AMBER)
    chip(3.87, y2, "V1-bis", "NOT DEMONSTRATED\nSTOP fixed 0/72k", RED)
    chip(5.73, y2, "S2b", "KILL −7.7 pp\nretention PASS", RED)
    chip(7.59, y2, "P2", "INDETERMINATE\n+34 pp not repl.", RED)

    y3 = 1.0
    chip(0.15, y3, "S5", "44/44 · 40/40\nstub Jev", GREEN)
    chip(2.01, y3, "Compiler v2", "100 % live\n504/504 · v1 5.86 %", GREEN)
    chip(3.87, y3, "E2E browser", "mechanics proven\nrefusal expected", AMBER)
    box(ax, 5.73, y3, 4.05, 1.75, "open locks\n(1) strict composition\n(2) depth via short search\n(3) web executor corpus + LINK",
        fc="#f5f3ff", ec=PURPLE, fs=8, weight="bold")
    ax.text(5.0, 0.28, "green = established · amber = exploratory/partial · red = not established / killed",
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

    # (d) P2 auxiliary-target ablation, v02 (final state)
    ax = axes[1, 1]
    labels = ["A\nimitation", "B\neffects\ncorrect", "D\nsimple\ntarget"]
    vals = [62.2, 66.3, 71.5]
    cols = [GREY, BLUE, GREEN]
    ax.bar(labels, vals, color=cols, alpha=0.88, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9, weight="bold")
    ax.set_ylim(0, 88); ax.set_ylabel("closed-loop success (%, both families)")
    ax.set_title("(d) Auxiliary-target ablation — v02 (non-confirmatory)", fontsize=9.5, weight="bold")
    ax.text(1.0, 80, "D−A = +9.4 pp: frozen bootstrap CI excl. 0, t/z FAIL\n"
                     "p = 0.047 one-sided (saturated family) · validity not in canon",
            ha="center", fontsize=7, color=RED)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    fig.suptitle("UCM results overview — statuses, instruments and negatives", fontsize=12, weight="bold", y=0.995)
    fig.text(0.99, 0.005, dpi_note, ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    save(fig, "fig6-results.svg")


# ---------------------------------------------------------------- fig 7: web compiler
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    # left: overall coverage v1 vs v2
    ax = axes[0]
    bars = ax.bar(["v1 probe\n(static)", "v2 snapshots", "v2 live refetch"],
                  [5.86, 100.0, 100.0], color=[RED, GREEN, GREEN], width=0.55)
    for b, v in zip(bars, [5.86, 100.0, 100.0]):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%" if v < 100 else "100%",
                ha="center", fontsize=10, weight="bold")
    ax.axhline(90, color=MUTED, ls="--", lw=0.9)
    ax.text(-0.42, 82.5, "GO bar 90%", color=MUTED, fontsize=8, ha="left")
    ax.set_ylim(0, 112); ax.set_ylabel("actionable-element coverage (%)")
    ax.set_title("(a) Compiler coverage", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    # right: per-page v1 vs v2
    ax = axes[1]
    pages = ["example", "httpbin", "httpbin-post", "w3schools"]
    v1 = [0, 0, 100, 5.6]
    v2 = [100, 100, 100, 100]
    x = np.arange(len(pages)) * 1.0
    w = 0.36
    ax.bar(x - w / 2, v1, w, label="v1 probe", color=RED, alpha=0.85)
    ax.bar(x + w / 2, v2, w, label="v2 (live)", color=GREEN, alpha=0.85)
    for xi, v in zip(x - w / 2, v1):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=7.5, color=RED)
    for xi, v in zip(x + w / 2, v2):
        ax.text(xi, v + 2, "100%", ha="center", fontsize=7.5, color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(pages, fontsize=8)
    ax.set_ylim(0, 135); ax.set_ylabel("coverage per page (%)")
    ax.set_title("(b) Per-page coverage (v1 vs v2)", fontsize=10, weight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95); ax.grid(axis="y", alpha=0.25, lw=0.5)

    fig.suptitle("The software bridge: actionability definition frozen from v1, parity verified, live-validated",
                 fontsize=10.5, weight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig7-web-compiler.svg")


# ---------------------------------------------------------------- fig 8: e2e chain
def fig8():
    fig, ax = blank_axes((9.8, 4.2))
    ax.set_xlim(0, 10.8)
    steps = [
        (0.15, "Browser\n(daemon + extension)\nlive DOM"),
        (2.10, "WebBridge\nsnapshot\n(HTML, tab)"),
        (4.05, "Compiler v2\nweb/2.0 typed\npolicy_input\n(link entities)"),
        (6.00, "UCM s5-full\nshim link→button\ntensorize SIW\n11.2 ms/decision"),
        (7.95, "Bridge\ncandidate → tool\nclick / fill /\nnavigate"),
    ]
    for x, label in steps:
        box(ax, x, 6.2, 1.85, 1.9, label, fc="#eef2ff" if x < 4 else "#ecfdf5", ec=BLUE if x < 4 else GREEN, fs=7.5)
    for x in (2.0, 3.95, 5.9, 7.85):
        arrow(ax, (x, 7.15), (x + 0.1, 7.15), color=MUTED)
    box(ax, 0.15, 3.4, 4.3, 1.7,
        "Smoke 1 — real page\n31 actionables · VIEW already satisfied\n→ STOP correct (11.2 ms)",
        fc="#fefce8", ec=AMBER, fs=8)
    box(ax, 4.9, 3.4, 5.5, 1.7,
        "Smoke 2 — real httpbin form\n4 actionables · 3 fields compiled (custtel, custemail, comments)\ninjected SET goal → STOP = calibrated refusal",
        fc="#fef2f2", ec=RED, fs=8)
    box(ax, 0.15, 1.0, 10.25, 1.6,
        "Mechanics proven: compile → decide → execute on a real browser.\n"
        "Open piece: an execution corpus of compiled pages\n"
        "(the same corpus is the training set and the benchmark).",
        fc="#f5f3ff", ec=PURPLE, fs=8.5, weight="bold")
    ax.text(5.0, 5.6, "documented shims: link→button · SELECT via option click · NAVIGATE via tool",
            ha="center", fontsize=8, color=MUTED, style="italic")
    save(fig, "fig8-e2e-chain.svg")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print("done")
