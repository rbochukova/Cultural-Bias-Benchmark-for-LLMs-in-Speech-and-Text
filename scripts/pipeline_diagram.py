"""
Generate the data aggregation pipeline diagram for the thesis:
"Measuring Cultural Bias in Multilingual LLMs Across Text and Speech"
Saves to notebooks/pipeline_diagram.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")

# ── Colour palette ──────────────────────────────────────────────────────────
C_SRC   = "#f0f0f0"   # data source boxes  (light grey)
C_STEP  = "#cfe2f3"   # processing steps   (light blue)
C_OBJ   = "#ffffff"   # intermediate objects (white, bold border)
C_OUT   = "#fce5cd"   # final output        (orange-ish)
C_EDGE  = "#333333"

def box(ax, x, y, w, h, label, color, fontsize=9, bold=False,
        radius=0.25, linewidth=1.5):
    """Draw a rounded rectangle with centred label."""
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor=C_EDGE, linewidth=linewidth,
        zorder=3
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, wrap=True, zorder=4,
            multialignment="center")

def arrow(ax, x0, y0, x1, y1, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE,
                                lw=1.4, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my+0.13, label, ha="center", va="bottom",
                fontsize=7.5, color="#555555", zorder=5)

def plus(ax, x, y, r=0.22):
    circle = plt.Circle((x, y), r, color="white", ec=C_EDGE,
                         linewidth=1.4, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, "+", ha="center", va="center", fontsize=13,
            fontweight="bold", zorder=4)

# ════════════════════════════════════════════════════════════════════════════
# COLUMN X positions
# ════════════════════════════════════════════════════════════════════════════
X_SRC  = 1.8    # data sources
X_S1   = 4.5    # processing step 1
X_S2   = 7.4    # processing step 2
X_OBJ  = 10.4   # intermediate objects
X_PLUS = 12.7   # merge (+) node
X_FINAL= 14.8   # final thesis object
X_RQ   = 17.1   # research questions

# ════════════════════════════════════════════════════════════════════════════
# ROW Y positions  (top → bottom)
# ════════════════════════════════════════════════════════════════════════════
Y1 = 9.0   # FLEURS / speech pipeline
Y2 = 6.6   # Text benchmarks — oracle path
Y3 = 4.6   # Text benchmarks — ASR path
Y4 = 2.3   # Speaker demographics

# ════════════════════════════════════════════════════════════════════════════
# 1 · FLEURS speech pipeline
# ════════════════════════════════════════════════════════════════════════════
box(ax, X_SRC, Y1,      2.8, 0.75, "FLEURS\n(google/fleurs)",        C_SRC, bold=True)
box(ax, X_S1,  Y1,      2.6, 0.65, "1.1  Filter\nEN · FR · BG",     C_STEP)
box(ax, X_S2,  Y1+0.5,  2.6, 0.65, "1.2  Whisper ASR\n(large-v3)",  C_STEP)
box(ax, X_S2,  Y1-0.5,  2.6, 0.65, "1.3  Oracle transcripts\n(WER = 0)", C_STEP)

arrow(ax, X_SRC+1.4, Y1,       X_S1-1.3, Y1)
arrow(ax, X_S1+1.3,  Y1,       X_S2-1.3, Y1+0.5)
arrow(ax, X_S1+1.3,  Y1,       X_S2-1.3, Y1-0.5)

box(ax, X_OBJ, Y1+0.5, 2.6, 0.65,
    "ASR hypotheses\n+ WER / CER", C_OBJ, bold=True, linewidth=2)
box(ax, X_OBJ, Y1-0.5, 2.6, 0.65,
    "Oracle references\nC2 baseline", C_OBJ, bold=True, linewidth=2)

arrow(ax, X_S2+1.3, Y1+0.5, X_OBJ-1.3, Y1+0.5)
arrow(ax, X_S2+1.3, Y1-0.5, X_OBJ-1.3, Y1-0.5)

# ════════════════════════════════════════════════════════════════════════════
# 2 · Text bias benchmarks — oracle BiasScore
# ════════════════════════════════════════════════════════════════════════════
box(ax, X_SRC, Y2, 2.8, 1.7,
    "Text benchmarks\n─────────────\nStereoSet (EN)\nCrowS-Pairs EN · FR\nEuroGEST (EN·FR·BG)\nSHADES (EN · FR)",
    C_SRC, bold=False, fontsize=8)

box(ax, X_S1, Y2,      2.6, 0.65, "2.1  SCM mapping\n(warmth / competence)", C_STEP)
box(ax, X_S2, Y2+0.4,  2.6, 0.65, "2.2  BiasScore\n(oracle / C2)",            C_STEP)
box(ax, X_S2, Y2-0.4,  2.6, 0.65, "2.3  BiasScore\n(ASR / C3)",               C_STEP)

arrow(ax, X_SRC+1.4, Y2,       X_S1-1.3, Y2)
arrow(ax, X_S1+1.3,  Y2,       X_S2-1.3, Y2+0.4)
arrow(ax, X_S1+1.3,  Y2,       X_S2-1.3, Y2-0.4)

# ASR hypotheses feed into C3 path
arrow(ax, X_OBJ-1.3, Y1+0.5, X_S2-1.3, Y2-0.4,
      label="hypotheses\n→ C3")

box(ax, X_OBJ, Y2, 2.6, 0.65,
    "Δ ASR object\nBiasScore(C3) − BiasScore(C2)",
    C_OBJ, bold=True, linewidth=2, fontsize=8.5)

arrow(ax, X_S2+1.3, Y2+0.4, X_OBJ-1.3, Y2+0.15)
arrow(ax, X_S2+1.3, Y2-0.4, X_OBJ-1.3, Y2-0.15)

# ════════════════════════════════════════════════════════════════════════════
# 3 · Speaker demographics
# ════════════════════════════════════════════════════════════════════════════
box(ax, X_SRC, Y4, 2.8, 1.3,
    "Speaker corpora\n──────────────\nMozilla SPS (EN · FR)\nbg_BG-dimitar (BG)",
    C_SRC, bold=False, fontsize=8)

box(ax, X_S1, Y4, 2.6, 0.65,
    "3.1  Sample &\ndemographic filter", C_STEP)

box(ax, X_OBJ, Y4, 2.6, 0.65,
    "Speaker demographics\ngender · age · accent",
    C_OBJ, bold=True, linewidth=2, fontsize=8.5)

arrow(ax, X_SRC+1.4, Y4, X_S1-1.3, Y4)
arrow(ax, X_S1+1.3,  Y4, X_OBJ-1.3, Y4)

# ════════════════════════════════════════════════════════════════════════════
# MERGE (+) and final objects
# ════════════════════════════════════════════════════════════════════════════
Y_MERGE = (Y1 + Y2) / 2   # ≈ 7.8
plus(ax, X_PLUS, Y_MERGE)

arrow(ax, X_OBJ+1.3, Y1+0.5,  X_PLUS-0.22, Y_MERGE+0.1)
arrow(ax, X_OBJ+1.3, Y1-0.5,  X_PLUS-0.22, Y_MERGE+0.05)
arrow(ax, X_OBJ+1.3, Y2,      X_PLUS-0.22, Y_MERGE-0.05)

box(ax, X_FINAL, Y_MERGE, 2.8, 0.75,
    "Benchmark dataset\nobject", C_OBJ, bold=True, linewidth=2)
arrow(ax, X_PLUS+0.22, Y_MERGE, X_FINAL-1.4, Y_MERGE)

# Demographics feeds into final object
arrow(ax, X_OBJ+1.3, Y4, X_FINAL-1.4, Y_MERGE-0.3)

# ════════════════════════════════════════════════════════════════════════════
# Research questions
# ════════════════════════════════════════════════════════════════════════════
arrow(ax, X_FINAL+1.4, Y_MERGE, X_RQ-0.9, Y_MERGE)

box(ax, X_RQ, Y1,   1.7, 0.65,
    "RQ1\nCultural bias\n(SCM)", C_OUT, fontsize=8.5)
box(ax, X_RQ, Y_MERGE, 1.7, 0.65,
    "RQ2\nΔ ASR\nattribution", C_OUT, fontsize=8.5)
box(ax, X_RQ, Y2,   1.7, 0.65,
    "RQ3\nError-type\nmechanism", C_OUT, fontsize=8.5)

arrow(ax, X_RQ-0.85, Y_MERGE, X_RQ-0.85+0.01, Y1+0.33)
arrow(ax, X_RQ-0.85, Y_MERGE, X_RQ-0.85+0.01, Y2+0.33)

# ════════════════════════════════════════════════════════════════════════════
# Title
# ════════════════════════════════════════════════════════════════════════════
ax.text(9, 10.6,
        "Data Aggregation Pipeline — Cultural Bias Benchmark for LLMs in Speech and Text",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#222222")

plt.tight_layout()
out = "c:/Users/user/Cultural-Bias-Benchmark-for-LLMs-in-Speech-and-Text/notebooks/pipeline_diagram.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
