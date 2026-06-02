#!/usr/bin/env python3
"""
Figure generation from existing figure_datasets/ only.
No recomputation. No data merging between control types.
"""

import os, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGDATA = os.path.join(OUTPUT_DIR, "results_final", "figure_datasets")
FIGURES = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ===========================================================================
# DATA LOADERS
# ===========================================================================
def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)

fig1_data = read_csv(os.path.join(FIGDATA, "fig1_consensus_ranking.csv"))
fig2_data = read_csv(os.path.join(FIGDATA, "fig2_control_distribution.csv"))
fig3_data = read_csv(os.path.join(FIGDATA, "fig3_zscore_heatmap.csv"))
fig4_data = read_csv(os.path.join(FIGDATA, "fig4_ips_variant_agreement.csv"))
fig5_data = read_csv(os.path.join(FIGDATA, "fig5_rank_stability.csv"))
fig5p_data = read_csv(os.path.join(FIGDATA, "fig5_rank_stability_primary.csv"))

# ===========================================================================
# HELPER: parse numeric safely
# ===========================================================================
def to_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

# ===========================================================================
# FIGURE 1: Final Consensus Ranking
# Control type: PRIMARY (P5_Scrambled, n=1, 30 contacts)
# Shows: P3 vs P1/P2/P4 vs Scrambled (primary control)
# ===========================================================================
candidates_f1 = [d for d in fig1_data if d["group"] == "candidate"]
control_f1 = [d for d in fig1_data if d["group"] == "control"]

fig1, ax1 = plt.subplots(figsize=(5.5, 4))
names = [d["peptide"] for d in fig1_data]
contacts = [to_float(d["contacts"]) for d in fig1_data]
density = [to_float(d["density"]) for d in fig1_data]

# Sort by contacts descending
order = sorted(range(len(fig1_data)), key=lambda i: contacts[i], reverse=True)
names = [names[i] for i in order]
contacts = [contacts[i] for i in order]

bar_colors = []
for n in names:
    if "P3" in n:
        bar_colors.append("#FF9800")
    elif "Scrambled" in n:
        bar_colors.append("#9E9E9E")
    else:
        bar_colors.append("#4A90D9")

x = np.arange(len(names))
bars = ax1.bar(x, contacts, color=bar_colors, edgecolor="black", linewidth=0.6, width=0.55)

# Annotate values
for i, (v, name) in enumerate(zip(contacts, names)):
    if "Scrambled" in name:
        label = f"{int(v)}"
    else:
        label = str(int(v))
    ax1.text(i, v + max(contacts) * 0.02, label,
             ha="center", va="bottom", fontsize=8, fontweight="bold")

# Highlight P3
for i, n in enumerate(names):
    if "P3" in n:
        bars[i].set_edgecolor("#E65100")
        bars[i].set_linewidth(2)

ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=7.5)
ax1.set_ylabel("Interface Contacts (count)", fontsize=9)
ax1.set_title("Final Consensus Ranking — Primary Control", fontsize=10, fontweight="bold")

# Footnote annotation
footnote_text = (
    "Control: primary (P5_Scrambled, n=1)\n"
    "Ensemble control not shown (see Fig2)"
)
ax1.text(0.99, 0.01, footnote_text, transform=ax1.transAxes,
         fontsize=6, color="0.4", ha="right", va="bottom",
         fontstyle="italic")

fig1.tight_layout()
fig1.savefig(os.path.join(FIGURES, "Fig1_final_consensus_ranking.png"))
plt.close(fig1)
print("  Fig1: Final consensus ranking (primary control only)")

# ===========================================================================
# FIGURE 2: Control Ensemble Distribution
# Control type: ENSEMBLE (5 calibration scrambles, all 0 contacts)
# Shows: distribution of scrambled controls with degenerate annotation
# ===========================================================================
# Fig2 data has 6 entries: primary scramble (30) + 5 calibration (0)
# Per rules: use ONLY the 5 calibration scrambles
calib_scrambles = [d for d in fig2_data if "Scrambled_" in d["peptide"]]
ctrl_contacts = [to_float(d["contacts"]) for d in calib_scrambles]

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(8, 3.5))

# Panel A: Scatter plot of ensemble values
ctrl_mean = np.mean(ctrl_contacts)
ctrl_std = np.std(ctrl_contacts, ddof=1)
ax2a.scatter(range(len(ctrl_contacts)), ctrl_contacts,
             color="#9E9E9E", s=60, zorder=5, edgecolor="black", linewidth=0.6)
# Candidates for overlay
cand_contacts_list = [to_float(d["contacts"]) for d in candidates_f1]
cand_names_list = [d["peptide"] for d in candidates_f1]
for i, (v, nm) in enumerate(zip(cand_contacts_list, cand_names_list)):
    color = "#FF9800" if "P3" in nm else "#4A90D9"
    offset = 0.15
    ax2a.scatter(-1 + offset * i, v, color=color, s=50, zorder=6,
                 edgecolor="black", linewidth=0.5, marker="s")
    ax2a.text(-1 + offset * i, v + 1.5, nm.split("_")[0],
              fontsize=5.5, ha="center", va="bottom", fontweight="bold")

ax2a.axhline(ctrl_mean, color="red", linestyle="--", linewidth=1,
             label=f"Ensemble mean = {ctrl_mean:.0f}")
ax2a.set_xlim(-1.5, len(ctrl_contacts) + 0.5)
ax2a.set_xticks(range(len(ctrl_contacts)))
ax2a.set_xticklabels([f"S{i+1}" for i in range(len(ctrl_contacts))], fontsize=7)
ax2a.set_ylabel("Interface Contacts (count)", fontsize=9)
ax2a.set_title("Ensemble Control Distribution", fontsize=9, fontweight="bold")
ax2a.legend(fontsize=7)
# Annotate degenerate
ax2a.text(0.5, 0.95, "DEGENERATE: all 0 contacts",
          transform=ax2a.transAxes, fontsize=7, color="red",
          ha="center", va="top", fontweight="bold",
          bbox=dict(facecolor="white", edgecolor="red", alpha=0.8, boxstyle="round,pad=0.3"))

# Panel B: Histogram
ax2b.hist(ctrl_contacts, bins=3, range=(-0.5, 2.5),
          color="#9E9E9E", edgecolor="black", linewidth=0.6, alpha=0.8)
ax2b.set_xlabel("Contacts", fontsize=9)
ax2b.set_ylabel("Count (scrambles)", fontsize=9)
ax2b.set_title("Contact Count Histogram (n=5)", fontsize=9, fontweight="bold")
ax2b.set_xticks([0])
ax2b.set_xlim(-0.5, 2.5)

fig2.suptitle("Control Ensemble: Calibration Scrambles (n=5)", fontsize=10, fontweight="bold", y=1.02)
footnote_text = (
    "Control: ensemble only (n=5, extended backbone)\n"
    "Distribution is degenerate under current contact threshold (3.5–4.5 A)\n"
    "All 5 scrambles produce 0 contacts → zero-variance baseline"
)
fig2.text(0.5, -0.02, footnote_text, fontsize=6.5, color="0.4",
          ha="center", fontstyle="italic", transform=fig2.transFigure)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGURES, "Fig2_control_ensemble_distribution.png"))
plt.close(fig2)
print("  Fig2: Control ensemble distribution (calibration only, degenerate annotated)")

# ===========================================================================
# FIGURE 3: Z-Score / Effect Size Heatmap
# Uses class labels (strong/weak), NOT raw z-values which are inflated by zero-variance.
# ===========================================================================
# Extract class labels for each peptide-metric pair
metrics_order = ["contacts", "density", "bbb_total"]
peptide_order_f3 = ["P3_Cyclized", "P4_RetroEnantio", "P2_Capped", "P1_Linear"]

class_matrix = np.empty((len(peptide_order_f3), len(metrics_order)), dtype=object)
# Also extract z-values for display (capped)
z_matrix = np.zeros((len(peptide_order_f3), len(metrics_order)))

for i, pep in enumerate(peptide_order_f3):
    for j, met in enumerate(metrics_order):
        class_key = f"class_{met}"
        z_key = f"z_{met}"
        # Find matching rows
        matches = [d for d in fig3_data if d["peptide"] == pep and d["metric"] == class_key]
        z_matches = [d for d in fig3_data if d["peptide"] == pep and d["metric"] == z_key]
        if matches:
            class_matrix[i, j] = matches[0]["value"]
        if z_matches:
            z_matrix[i, j] = to_float(z_matches[0]["value"])

# Cap z-values for display: anything > 10 → ">10"
z_display = np.empty_like(z_matrix, dtype=object)
for i in range(z_matrix.shape[0]):
    for j in range(z_matrix.shape[1]):
        v = z_matrix[i, j]
        if abs(v) > 10:
            z_display[i, j] = ">10" if v > 0 else "<-10"
        elif abs(v) < 0.01:
            z_display[i, j] = "~0"
        else:
            z_display[i, j] = f"{v:.1f}"

# Color map for class labels
cmap_class = {"strong": "#1B5E20", "moderate": "#FF8F00", "weak": "#B71C1C"}
cmap_bg = {"strong": "#C8E6C9", "moderate": "#FFF9C4", "weak": "#FFCDD2"}

fig3, ax3 = plt.subplots(figsize=(6, 3.5))
ax3.set_xlim(0, len(metrics_order))
ax3.set_ylim(0, len(peptide_order_f3))

cell_height = 1.0
cell_width = 1.0

for i in range(len(peptide_order_f3)):
    for j in range(len(metrics_order)):
        cls = class_matrix[i, j]
        zval = z_display[i, j]
        x = j + 0.5
        y = len(peptide_order_f3) - i - 0.5
        bg = cmap_bg.get(str(cls), "#FFFFFF")
        fc = cmap_class.get(str(cls), "#000000")
        rect = plt.Rectangle((j, len(peptide_order_f3) - i - 1),
                             cell_width, cell_height,
                             facecolor=bg, edgecolor="black", linewidth=0.5)
        ax3.add_patch(rect)
        # Class label
        ax3.text(x, y + 0.15, str(cls).upper(), ha="center", va="center",
                 fontsize=8, fontweight="bold", color=fc)
        # Capped z-value
        ax3.text(x, y - 0.2, f"Z={zval}", ha="center", va="center",
                 fontsize=6.5, color="0.3")

ax3.set_xticks(np.arange(len(metrics_order)) + 0.5)
ax3.set_xticklabels(["Contacts", "Density", "BBB"], fontsize=8)
ax3.set_yticks(np.arange(len(peptide_order_f3)) + 0.5)
ax3.set_yticklabels(peptide_order_f3, fontsize=7.5)
ax3.tick_params(axis="both", length=0)
ax3.set_xlabel("Metric", fontsize=9, labelpad=5)
ax3.set_ylabel("Peptide", fontsize=9, labelpad=5)
ax3.set_title("Effect Size Classification Heatmap", fontsize=10, fontweight="bold")
ax3.set_xlim(0, len(metrics_order))
ax3.set_ylim(0, len(peptide_order_f3))

# Legend
legend_patches = [
    mpatches.Patch(color=cmap_bg["strong"], label="Strong (|Z|>1)"),
    mpatches.Patch(color=cmap_bg["moderate"], label="Moderate (0.5<|Z|<1)"),
    mpatches.Patch(color=cmap_bg["weak"], label="Weak (|Z|<0.5)"),
]
ax3.legend(handles=legend_patches, loc="lower left", fontsize=6.5,
           bbox_to_anchor=(1.01, 0), frameon=True)

footnote = (
    "Z-scores relative to ensemble control (n=5, zero-variance)\n"
    "Z > 10 capped; numeric values not interpretable as standard deviations\n"
    "Classification: strong (|d|>1), moderate (0.5<|d|<1), weak (|d|<0.5)"
)
fig3.text(0.5, -0.02, footnote, fontsize=6, color="0.4",
          ha="center", fontstyle="italic", transform=fig3.transFigure)

fig3.tight_layout()
fig3.savefig(os.path.join(FIGURES, "Fig3_effect_size_heatmap.png"))
plt.close(fig3)
print("  Fig3: Effect size heatmap (capped z-values, class labels)")

# ===========================================================================
# FIGURE 4: IPS Model Agreement Plot
# Control type: PRIMARY for ranking, ENSEMBLE for robustness (separate panels)
# ===========================================================================
# Split fig4 into primary candidates only (no control)
cand_f4 = [d for d in fig4_data if d["peptide"] in ["P3_Cyclized","P1_Linear","P2_Capped","P4_RetroEnantio"]]

# Sort by consensus rank
cand_f4_sorted = sorted(cand_f4, key=lambda d: to_float(d["consensus_rank"]))

fig4, ((ax4a, ax4b), (ax4c, ax4d)) = plt.subplots(2, 2, figsize=(8, 6))

# Panel A: Bar comparison of 3 IPS variants
x = np.arange(len(cand_f4_sorted))
w = 0.22
variants = [
    ("IPS_original", "Original", ""),
    ("IPS_balanced", "Balanced", "//"),
    ("IPS_interface", "Interface-Dom.", "xx"),
]
variant_colors = ["#2196F3", "#4CAF50", "#FF9800"]

for idx, (key, label, hatch) in enumerate(variants):
    vals = [to_float(d[key]) for d in cand_f4_sorted]
    offset = (idx - 1) * w
    bars = ax4a.bar(x + offset, vals, w, label=label,
                    color=variant_colors[idx], edgecolor="black",
                    linewidth=0.5, hatch=hatch, alpha=0.85)
ax4a.set_xticks(x)
ax4a.set_xticklabels([d["peptide"] for d in cand_f4_sorted], rotation=15, ha="right", fontsize=7)
ax4a.set_ylabel("IPS Score", fontsize=8)
ax4a.set_title("3 Scoring Variants Comparison", fontsize=9, fontweight="bold")
ax4a.legend(fontsize=6)
ax4a.set_ylim(0, 1.05)
# Rank labels
for i, d in enumerate(cand_f4_sorted):
    ax4a.text(i, 0.0, f"#{d['consensus_rank']}", ha="center", va="bottom",
             fontsize=8, fontweight="bold", color="0.2")

# Panel B: Rank consistency matrix
ax4b.axis("off")
rank_data = []
for d in cand_f4_sorted:
    rank_data.append([
        d["peptide"],
        int(to_float(d["rank_IPS_original"])),
        int(to_float(d["rank_IPS_balanced"])),
        int(to_float(d["rank_IPS_interface"])),
        int(to_float(d["consensus_rank"])),
    ])

col_labels = ["Peptide", "Original", "Balanced", "Interface", "Consensus"]
table = ax4b.table(cellText=rank_data, colLabels=col_labels,
                   loc="center", cellLoc="center", fontsize=7)
table.auto_set_column_width(col=list(range(len(col_labels))))
table.auto_set_font_size(False)
table.set_fontsize(7)
for key, cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.5)
    if key[0] == 0:
        cell.set_facecolor("#E3F2FD")
        cell.set_text_props(fontweight="bold")
ax4b.set_title("Rank Consistency Across Models", fontsize=9, fontweight="bold")

# Panel C: Primary rank from pipeline.py (ESMFold)
f5p_sorted = sorted(fig5p_data, key=lambda d: to_float(d["mean_rank_primary"]))
names_5p = [d["peptide"] for d in f5p_sorted]
ranks_5p = [to_float(d["mean_rank_primary"]) for d in f5p_sorted]
colors_5p = ["#FF9800" if "P3" in d["peptide"] else "#4A90D9" for d in f5p_sorted]
bars = ax4c.barh(range(len(names_5p)), ranks_5p, color=colors_5p,
                 edgecolor="black", linewidth=0.6)
ax4c.set_yticks(range(len(names_5p)))
ax4c.set_yticklabels(names_5p, fontsize=7)
ax4c.invert_yaxis()
ax4c.set_xlabel("Rank (1=best)", fontsize=8)
ax4c.set_title("Primary Pipeline Rank (ESMFold)", fontsize=9, fontweight="bold")
ax4c.set_xticks([1, 2, 3, 4, 5])
for i, (v, n) in enumerate(zip(ranks_5p, names_5p)):
    ax4c.text(v + 0.1, i, f"#{int(v)}", va="center", fontsize=7, fontweight="bold")
ax4c.text(0.95, 0.05, "Control: primary (n=1)", transform=ax4c.transAxes,
          fontsize=6, color="0.4", ha="right", fontstyle="italic")

# Panel D: Calibration rank stability (ensemble controls)
f5_sorted = sorted(fig5_data, key=lambda d: to_float(d["mean_rank"]))
names_f5 = [d["peptide"] for d in f5_sorted]
mean_r = [to_float(d["mean_rank"]) for d in f5_sorted]
std_r = [to_float(d["std_rank"]) for d in f5_sorted]
min_r = [to_float(d["min_rank"]) for d in f5_sorted]
max_r = [to_float(d["max_rank"]) for d in f5_sorted]
colors_f5 = []
for nm in names_f5:
    if "P3" in nm:
        colors_f5.append("#FF9800")
    elif "Scrambled" in nm:
        colors_f5.append("#9E9E9E")
    else:
        colors_f5.append("#4A90D9")

for i in range(len(names_f5)):
    mn = min_r[i]
    mx = max_r[i]
    ax4d.plot([mn, mx], [i, i], color=colors_f5[i], linewidth=2, alpha=0.7)
    ax4d.scatter(mean_r[i], i, color=colors_f5[i], s=40, zorder=5,
                 edgecolor="black", linewidth=0.5)
    ax4d.text(mean_r[i] - 0.2, i, f"{int(mn)}-{int(mx)}",
              ha="right", va="center", fontsize=5.5, color="0.3")

ax4d.set_yticks(range(len(names_f5)))
ax4d.set_yticklabels(names_f5, fontsize=6)
ax4d.set_xlabel("Rank Range", fontsize=8)
ax4d.set_title("Calibration: Threshold Sensitivity (n=5 ensemble)", fontsize=9, fontweight="bold")
ax4d.invert_yaxis()
ax4d.text(0.95, 0.05, "Control: ensemble (n=5)", transform=ax4d.transAxes,
          fontsize=6, color="0.4", ha="right", fontstyle="italic")

fig4.suptitle("IPS Model Agreement & Rank Stability", fontsize=10, fontweight="bold", y=1.01)
footnote_f4 = (
    "Primary control used for ranking panels (left); "
    "Ensemble control used for robustness panels (right)\n"
    "Controls are NOT merged. "
    "Zero-variance ensemble produces degenerate rank ranges (all CV=0)"
)
fig4.text(0.5, -0.01, footnote_f4, fontsize=6, color="0.4",
          ha="center", fontstyle="italic", transform=fig4.transFigure)
fig4.tight_layout()
fig4.savefig(os.path.join(FIGURES, "Fig4_IPS_model_agreement.png"))
plt.close(fig4)
print("  Fig4: IPS model agreement (primary + ensemble panels separated)")

# ===========================================================================
# FIGURE 5: Stability Analysis Plot
# Primary for ranking comparison, ensemble for variance
# ===========================================================================
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(8, 3.5))

# Panel A: Rank stability across 125 threshold perturbations
f5_cands = [d for d in fig5_data if "Scrambled_" not in d["peptide"]]
names_f5a = [d["peptide"] for d in f5_cands]
mean_r5a = [to_float(d["mean_rank"]) for d in f5_cands]
std_r5a = [to_float(d["std_rank"]) for d in f5_cands]
min_r5a = [to_float(d["min_rank"]) for d in f5_cands]
max_r5a = [to_float(d["max_rank"]) for d in f5_cands]

x5a = np.arange(len(names_f5a))
colors_f5a = ["#FF9800" if "P3" in n else "#4A90D9" for n in names_f5a]
bars = ax5a.bar(x5a, max_r5a, color=colors_f5a, edgecolor="black", linewidth=0.6, alpha=0.3)
# Overlay mean as solid bar
ax5a.bar(x5a, mean_r5a, color=colors_f5a, edgecolor="black", linewidth=0.6)
# Add range lines
for i in range(len(names_f5a)):
    ax5a.plot([i, i], [min_r5a[i], max_r5a[i]], color="black", linewidth=1.5)
    ax5a.scatter(i, min_r5a[i], color="white", s=30, zorder=6, edgecolor="black", linewidth=0.5)
    ax5a.scatter(i, max_r5a[i], color="white", s=30, zorder=6, edgecolor="black", linewidth=0.5)
    ax5a.text(i, max_r5a[i] + 0.2, f"{int(min_r5a[i])}-{int(max_r5a[i])}",
              ha="center", fontsize=6, color="0.3")

ax5a.set_xticks(x5a)
ax5a.set_xticklabels(names_f5a, rotation=15, ha="right", fontsize=7.5)
ax5a.set_ylabel("Rank", fontsize=9)
ax5a.set_title("Rank Stability (125 threshold combos)", fontsize=9, fontweight="bold")
ax5a.set_ylim(0, max(max_r5a) + 1.5)
ax5a.invert_yaxis()
ax5a.text(0.95, 0.95, "All CV=0 (perfect stability)", transform=ax5a.transAxes,
          fontsize=6.5, color="red", ha="right", va="top", fontstyle="italic",
          bbox=dict(facecolor="white", edgecolor="red", alpha=0.7, boxstyle="round,pad=0.3"))

# Panel B: Seed perturbation comparison (seed 42 vs seed 99)
# From calibration data: seed 42 shows scores, seed 99 available in calibration outputs
# Plot the primary rank vs calibration stable rank
names_f5b = [d["peptide"] for d in f5p_sorted]
primary_ranks = [to_float(d["mean_rank_primary"]) for d in f5p_sorted]
# Calibration ranks (from mean_rank in fig5 for candidates)
calib_rank_map = {d["peptide"]: to_float(d["mean_rank"]) for d in f5_cands}
calib_ranks_5b = [calib_rank_map.get(n, 0) for n in names_f5b]

x5b = np.arange(len(names_f5b))
w5b = 0.3
ax5b.bar(x5b - w5b/2, primary_ranks, w5b, label="Primary (ESMFold)",
         color="#2196F3", edgecolor="black", linewidth=0.5)
ax5b.bar(x5b + w5b/2, calib_ranks_5b, w5b, label="Calibration (ext. backbone)",
         color="#4CAF50", edgecolor="black", linewidth=0.5, alpha=0.8)
ax5b.set_xticks(x5b)
ax5b.set_xticklabels(names_f5b, rotation=15, ha="right", fontsize=7.5)
ax5b.set_ylabel("Rank (1=best)", fontsize=9)
ax5b.set_title("Rank: Primary vs Calibration", fontsize=9, fontweight="bold")
ax5b.legend(fontsize=6.5)
ax5b.invert_yaxis()

fig5.suptitle("Rank Stability Across Methods & Perturbations", fontsize=10, fontweight="bold", y=1.02)
footnote_f5 = (
    "Left: 125 threshold combos (HB ±0.5, Hy ±0.5, SB ±0.5 in 0.25 steps)\n"
    "Right: Primary (ESMFold) vs Calibration (extended backbone, n=5 ensemble control)\n"
    "P3 rank is perfectly stable across all conditions"
)
fig5.text(0.5, -0.02, footnote_f5, fontsize=6.5, color="0.4",
          ha="center", fontstyle="italic", transform=fig5.transFigure)
fig5.tight_layout()
fig5.savefig(os.path.join(FIGURES, "Fig5_rank_stability.png"))
plt.close(fig5)
print("  Fig5: Rank stability (primary + calibration comparison)")

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 60)
print("FIGURE GENERATION COMPLETE")
print("=" * 60)
print(f"\n{FIGURES}/")
for f in sorted(os.listdir(FIGURES)):
    sz = os.path.getsize(os.path.join(FIGURES, f))
    print(f"  {f}: {sz//1024} KB")
print("\nControl usage per figure:")
print("  Fig1: Primary control only (P5_Scrambled, n=1, 30 contacts)")
print("  Fig2: Ensemble control only (5 calibration scrambles, all 0 contacts)")
print("  Fig3: Ensemble control for Z-scores (capped, >10 threshold)")
print("  Fig4: Primary for ranking (panels A,B,C); Ensemble for robustness (panel D)")
print("  Fig5: Primary for method comparison (panel B); Ensemble for threshold stability (panel A)")
print("\nNo statistical merging of control types was performed.")
