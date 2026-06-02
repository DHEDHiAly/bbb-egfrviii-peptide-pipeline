#!/usr/bin/env python3
"""Generate extended figures — on-figure labels OK, no bottom captions."""

import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
                      "savefig.facecolor": "white", "axes.spines.top": False,
                      "axes.spines.right": False})

def fl(v, d=0.0):
    try: return float(v)
    except: return d

def parse_atoms(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM"):
                ch = line[21].strip()
                rn = int(line[22:26].strip())
                rs = line[17:20].strip()
                an = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                out.append((ch, rn, rs, an, x, y, z))
    return out

def parse_ca(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ch = line[21].strip()
                rn = int(line[22:26].strip())
                rs = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                out.append((ch, rn, rs, x, y, z))
    return out

# ---- Fig6: Docking Pose --------------------------------------------------
def fig6():
    p = os.path.join(ROOT, "P3_Cyclized_complex.pdb")
    if not os.path.exists(p): return
    atm = parse_atoms(p)
    rca = np.array([[a[4],a[5],a[6]] for a in atm if a[0]=="A" and a[3]=="CA"])
    pca = np.array([[a[4],a[5],a[6]] for a in atm if a[0]=="B" and a[3]=="CA"])
    rec_info = [(a[1], a[2]) for a in atm if a[0]=="A" and a[3]=="CA"]
    if len(rca)==0 or len(pca)==0: return
    dst = np.min(np.linalg.norm(rca[:,None]-pca[None,:], axis=2), axis=1)
    cix = np.where(dst < 8.0)[0]

    fig6 = plt.figure(figsize=(6, 5.5))
    ax = fig6.add_subplot(111, projection="3d")
    ax.plot(rca[:,0], rca[:,1], rca[:,2], color="0.6", linewidth=0.5, alpha=0.5, label="Receptor")
    ax.scatter(rca[:,0], rca[:,1], rca[:,2], color="0.6", s=3, alpha=0.4)
    if len(cix) > 0:
        ax.scatter(rca[cix,0], rca[cix,1], rca[cix,2], color="#E53935", s=40,
                   edgecolor="black", linewidth=0.5, label="Contact residues")
    ax.plot(pca[:,0], pca[:,1], pca[:,2], color="#FF9800", linewidth=2.5, alpha=0.9, label="P3_Cyclized")
    ax.scatter(pca[:,0], pca[:,1], pca[:,2], color="#FF9800", s=30, edgecolor="black", linewidth=0.5)
    for idx in np.argsort(dst)[:5]:
        d2 = np.linalg.norm(pca - rca[idx], axis=1)
        pi = np.argmin(d2)
        ax.plot([rca[idx,0], pca[pi,0]], [rca[idx,1], pca[pi,1]],
                [rca[idx,2], pca[pi,2]], color="red", linewidth=0.5, linestyle="--", alpha=0.6)
    # Label closest receptor residue
    ci = np.argmin(dst)
    rn, rs = rec_info[ci]
    ax.text(rca[ci,0], rca[ci,1], rca[ci,2]+2, f"{rs}{rn}", fontsize=7, color="#E53935",
            fontweight="bold", ha="center")
    ax.text(pca[0,0], pca[0,1], pca[0,2], "N", fontsize=8, color="#FF9800", fontweight="bold", ha="center")
    ax.text(pca[-1,0], pca[-1,1], pca[-1,2], "C", fontsize=8, color="#FF9800", fontweight="bold", ha="center")
    ax.set_xlabel("X (A)", fontsize=7)
    ax.set_ylabel("Y (A)", fontsize=7)
    ax.set_zlabel("Z (A)", fontsize=7)
    ax.set_title("Docking Pose: P3 at EGFRvIII", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper right")
    fig6.savefig(os.path.join(FIG, "Fig6_docking_pose.png"))
    plt.close(fig6)

# ---- Fig7: BBB Radar -----------------------------------------------------
def fig7():
    st = []
    with open(os.path.join(ROOT, "results_final", "candidate_statistics.csv")) as f:
        for r in csv.DictReader(f): st.append(r)
    mp = {d["peptide"]: d for d in st}
    peps = ["P3_Cyclized","P4_RetroEnantio","P2_Capped","P1_Linear","P5_Scrambled"]
    crits = [("charge","Net Charge",2,8),("mw","MW (Da)",0,4000),
             ("gravy","GRAVY",-1,None),("instability","Instability",0,40)]
    n = len(crits)
    ang = np.linspace(0, 2*np.pi, n, endpoint=False).tolist() + [0]
    PAL = {"P3_Cyclized":"#FF9800","P4_RetroEnantio":"#4CAF50",
           "P2_Capped":"#2196F3","P1_Linear":"#9C27B0","P5_Scrambled":"#9E9E9E"}
    fig7, ax = plt.subplots(figsize=(5.5,5.5), subplot_kw=dict(polar=True))
    for pn in peps:
        if pn not in mp: continue
        d = mp[pn]; vs = []
        for k,_,lo,hi in crits:
            v = fl(d[k])
            if hi is not None:
                vs.append(max(0, min(1, (v-lo)/(hi-lo))) if hi!=lo else 0.5)
            elif lo is not None:
                vs.append(max(0, min(1, v/(v+1))))
            else: vs.append(0.5)
        vs += vs[:1]
        c = PAL.get(pn,"#999")
        ax.fill(ang, vs, alpha=0.08, color=c)
        ax.plot(ang, vs, "o-", color=c, linewidth=1.8, markersize=5, label=pn.replace("_"," "))
    ax.plot(ang, [0.5]*len(ang), "k--", linewidth=0.8, alpha=0.4, label="Threshold")
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([c[1] for c in crits], fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"], fontsize=6, color="0.4")
    ax.set_title("BBB Heuristic Criteria", fontsize=10, fontweight="bold", pad=20)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.15), fontsize=7, ncol=3)
    fig7.savefig(os.path.join(FIG, "Fig7_BBB_radar.png"))
    plt.close(fig7)

# ---- Fig8: Peptide Overlay -----------------------------------------------
def fig8():
    peps = ["P1_Linear","P2_Capped","P3_Cyclized","P4_RetroEnantio"]
    PAL = {"P1_Linear":"#9C27B0","P2_Capped":"#2196F3",
           "P3_Cyclized":"#FF9800","P4_RetroEnantio":"#4CAF50"}
    SEQS = {"P1_Linear":"TFFYGGSRGKRNNFKTEGWRGGRL",
            "P2_Capped":"TFFYGGSRGKRNNFKTEGWRGGRL",
            "P3_Cyclized":"TFFYGGSRGKRNNFKTCVPLPHLKFC",
            "P4_RetroEnantio":"TFFYGGSRGKRNNFKTEGWRGGRL"}
    fig8 = plt.figure(figsize=(6,5))
    ax = fig8.add_subplot(111, projection="3d")
    ref = None
    for pn in peps:
        pa = os.path.join(ROOT, f"{pn}.pdb")
        if not os.path.exists(pa): continue
        ca = parse_ca(pa)
        if not ca: continue
        co = np.array([[r[3],r[4],r[5]] for r in ca])
        if ref is None:
            ref = co[0].copy()
        else:
            co = co + (ref - co[0])
        ax.plot(co[:,0], co[:,1], co[:,2], "-o", color=PAL.get(pn,"#333"),
                linewidth=2, markersize=4, alpha=0.85, label=pn.replace("_"," "),
                markeredgecolor="black", markeredgewidth=0.3)
    ax.set_xlabel("X (A)", fontsize=7)
    ax.set_ylabel("Y (A)", fontsize=7)
    ax.set_zlabel("Z (A)", fontsize=7)
    ax.set_title("Peptide Structure Overlay", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    # Sequence annotation
    stxt = "Sequences:\n" + "\n".join(f"  {pn.split('_')[0]}: {SEQS[pn]}" for pn in peps)
    ax.text2D(0.02, 0.02, stxt, transform=ax.transAxes, fontsize=5.5, color="0.3",
              va="bottom", ha="left", fontfamily="monospace",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    fig8.savefig(os.path.join(FIG, "Fig8_peptide_overlay.png"))
    plt.close(fig8)

# ---- Fig9: Contact Breakdown ---------------------------------------------
def fig9():
    st = []
    with open(os.path.join(ROOT, "results_final", "candidate_statistics.csv")) as f:
        for r in csv.DictReader(f): st.append(r)
    mp = {d["peptide"]: d for d in st}
    order = ["P3_Cyclized","P4_RetroEnantio","P2_Capped","P1_Linear"]
    fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(8, 4.5))
    names = order + ["P5_Scrambled"]
    hb = [fl(mp[p]["hbonds"]) for p in order] + [fl(mp["P5_Scrambled"]["hbonds"])]
    hy = [fl(mp[p]["hydrophobic"]) for p in order] + [fl(mp["P5_Scrambled"]["hydrophobic"])]
    sb = [fl(mp[p]["salt_bridges"]) for p in order] + [fl(mp["P5_Scrambled"]["salt_bridges"])]
    x = np.arange(len(names)); w = 0.5
    ax9a.bar(x, hb, w, label="H-bonds", color="#43A047", edgecolor="black", linewidth=0.5)
    ax9a.bar(x, hy, w, bottom=hb, label="Hydrophobic", color="#FB8C00", edgecolor="black", linewidth=0.5)
    ax9a.bar(x, sb, w, bottom=[hb[i]+hy[i] for i in range(len(hb))],
             label="Salt bridges", color="#E53935", edgecolor="black", linewidth=0.5)
    for i in range(len(names)):
        t = hb[i]+hy[i]+sb[i]
        ax9a.text(i, t+1.5, str(int(t)), ha="center", fontsize=8, fontweight="bold")
    ax9a.set_xticks(x)
    ax9a.set_xticklabels(names, rotation=15, ha="right", fontsize=7.5)
    ax9a.set_ylabel("Contact count", fontsize=9)
    ax9a.set_title("Contact Breakdown", fontsize=9, fontweight="bold")
    ax9a.legend(fontsize=7)

    for i in range(len(names)):
        t = hb[i]+hy[i]+sb[i]
        hp = hb[i]/t*100 if t>0 else 0
        hyp = hy[i]/t*100 if t>0 else 0
        sbp = sb[i]/t*100 if t>0 else 0
        ax9b.bar(i, hp, w, color="#43A047", edgecolor="black", linewidth=0.5)
        ax9b.bar(i, hyp, w, bottom=hp, color="#FB8C00", edgecolor="black", linewidth=0.5)
        ax9b.bar(i, sbp, w, bottom=hp+hyp, color="#E53935", edgecolor="black", linewidth=0.5)
        if hyp > 20:
            ax9b.text(i, hp+hyp/2, f"{hyp:.0f}%", ha="center", fontsize=7, fontweight="bold", color="white")
        if hp > 15:
            ax9b.text(i, hp/2, f"{hp:.0f}%", ha="center", fontsize=7, fontweight="bold", color="white")
    ax9b.set_xticks(x)
    ax9b.set_xticklabels(names, rotation=15, ha="right", fontsize=7.5)
    ax9b.set_ylabel("Proportion (%)", fontsize=9)
    ax9b.set_title("Contact Proportion", fontsize=9, fontweight="bold")
    ax9b.set_ylim(0, 110)
    fig9.savefig(os.path.join(FIG, "Fig9_contact_breakdown.png"))
    plt.close(fig9)

if __name__ == "__main__":
    fig6(); fig7(); fig8(); fig9()
    print("Done: Fig6–Fig9")
