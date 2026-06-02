#!/usr/bin/env python3
"""Generate manuscript figures — no text overlay."""

import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(ROOT, "results_final", "figure_datasets")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
                      "savefig.facecolor": "white", "axes.spines.top": False,
                      "axes.spines.right": False})

def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))

def fl(v, d=0.0):
    try: return float(v)
    except: return d

# ---- Fig1 ---------------------------------------------------------------
d1 = read_csv(os.path.join(FD, "fig1_consensus_ranking.csv"))
names = [r["peptide"] for r in d1]
vals = [fl(r["contacts"]) for r in d1]
order = sorted(range(len(d1)), key=lambda i: vals[i], reverse=True)
names = [names[i] for i in order]
vals = [vals[i] for i in order]
colors = ["#FF9800" if "P3" in n else "#9E9E9E" if "Scrambled" in n else "#4A90D9" for n in names]

fig1, ax1 = plt.subplots(figsize=(5.5, 4))
ax1.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
ax1.set_xticks([])
ax1.set_yticks([])
for s in ["left","bottom"]:
    ax1.spines[s].set_visible(False)
fig1.savefig(os.path.join(FIG, "Fig1_final_consensus_ranking.png"))
plt.close(fig1)

# ---- Fig2 ---------------------------------------------------------------
d2 = read_csv(os.path.join(FD, "fig2_control_distribution.csv"))
cal = [d for d in d2 if "Scrambled_" in d["peptide"]]
cal_v = [fl(d["contacts"]) for d in cal]
ca1 = [fl(d["contacts"]) for d in d1 if d["group"] == "candidate"]
ca2 = [d["peptide"] for d in d1 if d["group"] == "candidate"]

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(8, 3.5))
ax2a.scatter(range(len(cal_v)), cal_v, color="#9E9E9E", s=60, edgecolor="black", linewidth=0.6)
for i, (v, nm) in enumerate(zip(ca1, ca2)):
    c = "#FF9800" if "P3" in nm else "#4A90D9"
    ax2a.scatter(-1 + 0.15*i, v, color=c, s=50, edgecolor="black", linewidth=0.5, marker="s")
ax2a.set_xticks([]); ax2a.set_yticks([])
for s in ["left","bottom"]: ax2a.spines[s].set_visible(False)

ax2b.hist(cal_v, bins=3, range=(-0.5, 2.5), color="#9E9E9E", edgecolor="black", linewidth=0.6)
ax2b.set_xticks([]); ax2b.set_yticks([])
for s in ["left","bottom"]: ax2b.spines[s].set_visible(False)
fig2.savefig(os.path.join(FIG, "Fig2_control_ensemble_distribution.png"))
plt.close(fig2)

# ---- Fig3 ---------------------------------------------------------------
d3 = read_csv(os.path.join(FD, "fig3_zscore_heatmap.csv"))
metrics = ["contacts","density","bbb_total"]
peps = ["P3_Cyclized","P4_RetroEnantio","P2_Capped","P1_Linear"]
cls = {"strong":"#C8E6C9","moderate":"#FFF9C4","weak":"#FFCDD2"}
fig3, ax3 = plt.subplots(figsize=(6, 3.5))
for i, p in enumerate(peps):
    for j, m in enumerate(metrics):
        row = [r for r in d3 if r["peptide"]==p and r["metric"]==f"class_{m}"]
        c = cls.get(str(row[0]["value"]) if row else "weak", "#FFF")
        ax3.add_patch(plt.Rectangle((j, len(peps)-i-1), 1, 1, facecolor=c, edgecolor="black", linewidth=0.5))
ax3.set_xlim(0, len(metrics)); ax3.set_ylim(0, len(peps))
ax3.set_xticks([]); ax3.set_yticks([])
for s in ["left","bottom"]: ax3.spines[s].set_visible(False)
fig3.savefig(os.path.join(FIG, "Fig3_effect_size_heatmap.png"))
plt.close(fig3)

# ---- Fig4 ---------------------------------------------------------------
d4 = read_csv(os.path.join(FD, "fig4_ips_variant_agreement.csv"))
d5p = read_csv(os.path.join(FD, "fig5_rank_stability_primary.csv"))
d5 = read_csv(os.path.join(FD, "fig5_rank_stability.csv"))
d4c = [d for d in d4 if d["peptide"] in ["P3_Cyclized","P1_Linear","P2_Capped","P4_RetroEnantio"]]
d4s = sorted(d4c, key=lambda d: fl(d["consensus_rank"]))
fig4, ((ax4a,ax4b),(ax4c,ax4d)) = plt.subplots(2,2,figsize=(8,6))
x = np.arange(len(d4s)); w = 0.22
for idx, key in enumerate(["IPS_original","IPS_balanced","IPS_interface"]):
    v = [fl(d[key]) for d in d4s]
    ax4a.bar(x + (idx-1)*w, v, w, color=["#2196F3","#4CAF50","#FF9800"][idx],
             edgecolor="black", linewidth=0.5, alpha=0.85)
ax4a.set_xticks([]); ax4a.set_yticks([])
for s in ["left","bottom"]: ax4a.spines[s].set_visible(False)

ax4b.axis("off")
rd = [[d["peptide"],int(fl(d["rank_IPS_original"])),int(fl(d["rank_IPS_balanced"])),
       int(fl(d["rank_IPS_interface"])),int(fl(d["consensus_rank"]))] for d in d4s]
t = ax4b.table(cellText=rd, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(7)
for k, c in t.get_celld().items():
    c.set_edgecolor("black"); c.set_linewidth(0.5)
    if k[0]==0: c.set_facecolor("#E3F2FD")

d5ps = sorted(d5p, key=lambda d: fl(d["mean_rank_primary"]))
ax4c.barh(range(len(d5ps)), [fl(d["mean_rank_primary"]) for d in d5ps],
          color=["#FF9800" if "P3" in d["peptide"] else "#4A90D9" for d in d5ps],
          edgecolor="black", linewidth=0.6)
ax4c.set_yticks([]); ax4c.set_xticks([])
for s in ["left","bottom"]: ax4c.spines[s].set_visible(False)

d5s = sorted(d5, key=lambda d: fl(d["mean_rank"]))
for i in range(len(d5s)):
    mn, mx = fl(d5s[i]["min_rank"]), fl(d5s[i]["max_rank"])
    c = "#FF9800" if "P3" in d5s[i]["peptide"] else "#9E9E9E" if "Scrambled" in d5s[i]["peptide"] else "#4A90D9"
    ax4d.plot([mn, mx], [i, i], color=c, linewidth=2, alpha=0.7)
    ax4d.scatter(fl(d5s[i]["mean_rank"]), i, color=c, s=40, edgecolor="black", linewidth=0.5)
ax4d.set_yticks([]); ax4d.set_xticks([])
for s in ["left","bottom"]: ax4d.spines[s].set_visible(False)
fig4.savefig(os.path.join(FIG, "Fig4_IPS_model_agreement.png"))
plt.close(fig4)

# ---- Fig5 ---------------------------------------------------------------
d5c = [d for d in d5 if "Scrambled_" not in d["peptide"]]
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(8, 3.5))
names5 = [d["peptide"] for d in d5c]
mean5 = [fl(d["mean_rank"]) for d in d5c]
min5 = [fl(d["min_rank"]) for d in d5c]
max5 = [fl(d["max_rank"]) for d in d5c]
colors5 = ["#FF9800" if "P3" in n else "#4A90D9" for n in names5]
ax5a.bar(range(len(names5)), max5, color=colors5, edgecolor="black", linewidth=0.6, alpha=0.3)
ax5a.bar(range(len(names5)), mean5, color=colors5, edgecolor="black", linewidth=0.6)
for i in range(len(names5)):
    ax5a.plot([i,i], [min5[i], max5[i]], color="black", linewidth=1.5)
    ax5a.scatter(i, min5[i], color="white", s=30, edgecolor="black", linewidth=0.5)
    ax5a.scatter(i, max5[i], color="white", s=30, edgecolor="black", linewidth=0.5)
ax5a.set_xticks([]); ax5a.set_yticks([])
for s in ["left","bottom"]: ax5a.spines[s].set_visible(False)

d5ps2 = sorted([d for d in d5p if d["peptide"] != "P5_Scrambled"], key=lambda d: fl(d["mean_rank_primary"]))
pr = [fl(d["mean_rank_primary"]) for d in d5ps2]
cr = [fl(d["mean_rank"]) for d in d5s if "Scrambled_" not in d["peptide"]]
x5 = np.arange(len(d5ps2)); w5 = 0.3
ax5b.bar(x5-w5/2, pr, w5, color="#2196F3", edgecolor="black", linewidth=0.5)
ax5b.bar(x5+w5/2, cr, w5, color="#4CAF50", edgecolor="black", linewidth=0.5, alpha=0.8)
ax5b.set_xticks([]); ax5b.set_yticks([])
for s in ["left","bottom"]: ax5b.spines[s].set_visible(False)
fig5.savefig(os.path.join(FIG, "Fig5_rank_stability.png"))
plt.close(fig5)

print("Done: Fig1–Fig5 generated in figures/")
