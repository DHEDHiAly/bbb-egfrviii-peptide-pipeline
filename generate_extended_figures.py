#!/usr/bin/env python3
"""
Generate 4 extended manuscript figures: docking pose, BBB radar,
peptide overlay, interface contact breakdown.
"""

import os, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(ROOT, "figures")
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
# DATA
# ===========================================================================
def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)

stats = read_csv(os.path.join(ROOT, "results_final", "candidate_statistics.csv"))
cand = [d for d in stats if d["group"] == "candidate"]
ctrl = [d for d in stats if d["group"] == "control"]

PEPTIDE_ORDER = ["P3_Cyclized", "P4_RetroEnantio", "P2_Capped", "P1_Linear"]
PALETTE = {"P3_Cyclized": "#FF9800", "P4_RetroEnantio": "#4CAF50",
           "P2_Capped": "#2196F3", "P1_Linear": "#9C27B0",
           "P5_Scrambled": "#9E9E9E"}
# Sequences from pipeline_summary.txt
SEQUENCES = {
    "P1_Linear": "TFFYGGSRGKRNNFKTEGWRGGRL",
    "P2_Capped": "TFFYGGSRGKRNNFKTEGWRGGRL",
    "P3_Cyclized": "TFFYGGSRGKRNNFKTCVPLPHLKFC",
    "P4_RetroEnantio": "TFFYGGSRGKRNNFKTEGWRGGRL",
}

def to_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

def ntof(v, default=0.0):
    return to_float(v, default)

# ===========================================================================
# PDB PARSER
# ===========================================================================
def parse_pdb_ca(path):
    """Extract (chain, resnum, resname, x, y, z) for all Cα atoms."""
    atoms = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain = line[21].strip()
                resname = line[17:20].strip()
                resnum = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append((chain, resnum, resname, x, y, z))
    return atoms

def parse_pdb_atoms(path):
    """Extract all atoms: (chain, resnum, resname, atomname, x, y, z)."""
    atoms = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21].strip()
                resname = line[17:20].strip()
                resnum = int(line[22:26].strip())
                atomname = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append((chain, resnum, resname, atomname, x, y, z))
    return atoms

# ===========================================================================
# FIGURE 6: DOCKING POSE — Receptor + P3 in binding site
# ===========================================================================
def fig6_docking_pose():
    print("  Fig6: Docking pose...")
    # Parse receptor (chain A) and peptide (chain B) from P3 complex
    complex_path = os.path.join(ROOT, "P3_Cyclized_complex.pdb")
    if not os.path.exists(complex_path):
        print("    SKIP: complex PDB not found")
        return
    atoms = parse_pdb_atoms(complex_path)
    rec_ca = [(r[1], r[2], r[4], r[5], r[6]) for r in atoms
              if r[0] == "A" and r[3] == "CA"]
    pep_ca = [(r[1], r[2], r[4], r[5], r[6]) for r in atoms
              if r[0] == "B" and r[3] == "CA"]

    if not rec_ca or not pep_ca:
        print("    SKIP: no Cα atoms found")
        return

    # Compute distances from each receptor Cα to nearest peptide Cα
    rec_coords = np.array([[r[2], r[3], r[4]] for r in rec_ca])
    pep_coords = np.array([[r[2], r[3], r[4]] for r in pep_ca])
    dists = np.min(np.linalg.norm(rec_coords[:, None] - pep_coords[None, :], axis=2), axis=1)

    # Identify contact residues (within 8 A)
    contact_idx = np.where(dists < 8.0)[0]
    close_idx = np.argsort(dists)[:10]  # 10 closest

    fig6 = plt.figure(figsize=(6, 5.5))
    ax = fig6.add_subplot(111, projection="3d")

    # Receptor backbone trace
    ax.plot(rec_coords[:, 0], rec_coords[:, 1], rec_coords[:, 2],
            color="0.6", linewidth=0.5, alpha=0.5, label="Receptor Cα trace")
    ax.scatter(rec_coords[:, 0], rec_coords[:, 1], rec_coords[:, 2],
               color="0.6", s=3, alpha=0.4)

    # Receptor contact residues (within 8A of peptide)
    if len(contact_idx) > 0:
        contact_coords = rec_coords[contact_idx]
        ax.scatter(contact_coords[:, 0], contact_coords[:, 1], contact_coords[:, 2],
                   color="#E53935", s=40, zorder=6, edgecolor="black", linewidth=0.5,
                   label=f"Contact residues (n={len(contact_idx)})")

    # Peptide backbone
    ax.plot(pep_coords[:, 0], pep_coords[:, 1], pep_coords[:, 2],
            color="#FF9800", linewidth=2.5, alpha=0.9, label="P3_Cyclized")
    ax.scatter(pep_coords[:, 0], pep_coords[:, 1], pep_coords[:, 2],
               color="#FF9800", s=30, zorder=7, edgecolor="black", linewidth=0.5)

    # Dashed lines for 5 closest receptor-peptide Cα pairs
    closest_pep_idx = np.argmin(dists)
    for idx in close_idx[:5]:
        rec_pt = rec_coords[idx]
        # Find closest peptide Cα to this receptor residue
        d = np.linalg.norm(pep_coords - rec_pt, axis=1)
        pep_idx = np.argmin(d)
        ax.plot([rec_pt[0], pep_coords[pep_idx, 0]],
                [rec_pt[1], pep_coords[pep_idx, 1]],
                [rec_pt[2], pep_coords[pep_idx, 2]],
                color="red", linewidth=0.5, linestyle="--", alpha=0.6)

    # Label closest receptor residue
    if len(rec_ca) > 0:
        closest_idx = np.argmin(dists)
        resnum, resname = rec_ca[closest_idx][0], rec_ca[closest_idx][1]
        label = f"{resname}{resnum}"
        ax.text(rec_coords[closest_idx, 0], rec_coords[closest_idx, 1],
                rec_coords[closest_idx, 2] + 2.0, label,
                fontsize=7, color="#E53935", fontweight="bold",
                ha="center", va="bottom")

    # Label peptide N- and C-termini
    if len(pep_ca) > 0:
        ax.text(pep_coords[0, 0], pep_coords[0, 1], pep_coords[0, 2],
                "N", fontsize=8, color="#FF9800", fontweight="bold", ha="center")
        ax.text(pep_coords[-1, 0], pep_coords[-1, 1], pep_coords[-1, 2],
                "C", fontsize=8, color="#FF9800", fontweight="bold", ha="center")

    ax.set_xlabel("X (Å)", fontsize=7)
    ax.set_ylabel("Y (Å)", fontsize=7)
    ax.set_zlabel("Z (Å)", fontsize=7)
    ax.set_title("Docking Pose: P3_Cyclized at EGFRvIII Binding Site", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper right")

    # Footnote
    footnote = (
        "Receptor: EGFRvIII (PDB 8UKX, chain A). Peptide: P3_Cyclized (chain B).\n"
        "Red spheres: receptor Cα within 8Å of any peptide atom. Red dashed lines: closest Cα-Cα contacts.\n"
        "This is a rigid-body heuristic docking pose, not a minimized complex."
    )
    fig6.text(0.5, 0.01, footnote, fontsize=6, color="0.4",
              ha="center", fontstyle="italic", transform=fig6.transFigure)
    # Adjust for footnote
    fig6.subplots_adjust(bottom=0.12)
    fig6.savefig(os.path.join(FIGURES, "Fig6_docking_pose.png"))
    plt.close(fig6)
    print(f"    Done: receptor Cα={len(rec_ca)}, peptide Cα={len(pep_ca)}, contacts={len(contact_idx)}")

# ===========================================================================
# FIGURE 7: BBB HEURISTIC RADAR PLOT
# ===========================================================================
def fig7_bbb_radar():
    print("  Fig7: BBB heuristic radar plot...")

    # Extract data
    peptides_for_radar = PEPTIDE_ORDER + ["P5_Scrambled"]
    plot_data = {d["peptide"]: d for d in stats if d["peptide"] in peptides_for_radar}

    # 4 criteria with thresholds and display names
    criteria = [
        ("charge", "Net Charge\n(pH 7.4)", 2.0, 8.0, "Pass: 2–8"),
        ("mw", "MW\n(Da)", 0, 4000, "Pass: <4000"),
        ("gravy", "GRAVY\n(Hydrophobicity)", -1.0, None, "Pass: >−1.0"),
        ("instability", "Instability\nIndex", 0, 40, "Pass: <40"),
    ]

    n_crit = len(criteria)
    angles = np.linspace(0, 2 * np.pi, n_crit, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    fig7, ax7 = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))

    # Plot each peptide
    for pname in peptides_for_radar:
        if pname not in plot_data:
            continue
        d = plot_data[pname]
        vals = []
        for (key, _, lower, upper, _) in criteria:
            v = ntof(d[key])
            # Normalize to 0–1: 0=worst, 1=best for BBB
            if upper is not None and lower is not None:
                if upper != lower:
                    # Target range
                    if v < lower:
                        norm = 0.0
                    elif v > upper:
                        norm = 1.0
                    else:
                        norm = (v - lower) / (upper - lower)
                else:
                    norm = 0.5
            elif upper is not None:
                # Single upper bound
                norm = max(0, min(1, 1.0 - v / upper))
            elif lower is not None:
                # Single lower bound
                norm = max(0, min(1, v / (v + 1)))  # v relative to |lower|
            else:
                norm = 0.5
            vals.append(norm)
        vals += vals[:1]
        color = PALETTE.get(pname, "#9E9E9E")
        ax7.fill(angles, vals, alpha=0.08, color=color)
        ax7.plot(angles, vals, "o-", color=color, linewidth=1.8,
                 label=pname.replace("_", " "), markersize=5)

    # Add threshold markers
    threshold_vals = []
    for (key, _, lower, upper, _) in criteria:
        if upper is not None:
            threshold_vals.append(0.5)  # Midpoint as threshold indicator
        elif lower is not None:
            threshold_vals.append(0.5)
        else:
            threshold_vals.append(0.5)
    threshold_vals += threshold_vals[:1]
    ax7.plot(angles, threshold_vals, "k--", linewidth=0.8, alpha=0.4, label="Pass threshold")

    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels([c[1] for c in criteria], fontsize=7.5)
    ax7.set_ylim(0, 1)
    ax7.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax7.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6, color="0.4")
    ax7.set_title("BBB Heuristic Criteria — Normalized Scores", fontsize=10,
                  fontweight="bold", pad=20)

    # Add criteria annotations
    for i, (_, _, lower, upper, desc) in enumerate(criteria):
        ang = angles[i]
        ax7.text(ang, 1.15, desc, fontsize=6, color="0.4",
                 ha="center", va="center", rotation=np.degrees(ang))

    ax7.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.15),
               fontsize=7, ncol=3)

    footnote = (
        "Normalization: 1.0 = fully passes BBB criterion. "
        "Dashed circle = approximate pass threshold (0.5).\n"
        "BBB criteria are heuristic filters (charge 2–8, MW<4kDa, GRAVY>−1.0, instability<40). "
        "Not validated against experimental permeability."
    )
    fig7.text(0.5, 0.01, footnote, fontsize=6, color="0.4",
              ha="center", fontstyle="italic", transform=fig7.transFigure)
    fig7.subplots_adjust(bottom=0.12)
    fig7.savefig(os.path.join(FIGURES, "Fig7_BBB_radar.png"))
    plt.close(fig7)
    print("    Done")

# ===========================================================================
# FIGURE 8: PEPTIDE STRUCTURE OVERLAY
# ===========================================================================
def fig8_peptide_overlay():
    print("  Fig8: Peptide structure overlay...")
    fig8 = plt.figure(figsize=(6, 5))
    ax = fig8.add_subplot(111, projection="3d")

    peptide_pdbs = {
        "P1_Linear": os.path.join(ROOT, "P1_Linear.pdb"),
        "P2_Capped": os.path.join(ROOT, "P2_Capped.pdb"),
        "P3_Cyclized": os.path.join(ROOT, "P3_Cyclized.pdb"),
        "P4_RetroEnantio": os.path.join(ROOT, "P4_RetroEnantio.pdb"),
    }

    # Align by first Cα position
    ref_ca = None
    aligned = {}
    for pname in PEPTIDE_ORDER:
        path = peptide_pdbs.get(pname)
        if not path or not os.path.exists(path):
            print(f"    SKIP {pname}: PDB not found")
            continue
        ca_atoms = parse_pdb_ca(path)
        if not ca_atoms:
            print(f"    SKIP {pname}: no Cα atoms")
            continue
        coords = np.array([[r[3], r[4], r[5]] for r in ca_atoms])
        if ref_ca is None:
            ref_ca = coords[0].copy()
            offset = np.zeros(3)
        else:
            offset = ref_ca - coords[0]
            coords = coords + offset
        aligned[pname] = coords

    if not aligned:
        print("    ERROR: no peptide structures loaded")
        return

    for pname, coords in aligned.items():
        color = PALETTE.get(pname, "#333")
        label = pname.replace("_", " ")
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                "-o", color=color, linewidth=2, markersize=4,
                label=label, alpha=0.85, markerfacecolor=color,
                markeredgecolor="black", markeredgewidth=0.3)

        # Label N and C
        ax.text(coords[0, 0], coords[0, 1], coords[0, 2] + 1.5,
                f"{pname.split('_')[0]} N", fontsize=6, color=color,
                fontweight="bold", ha="center")
        ax.text(coords[-1, 0], coords[-1, 1], coords[-1, 2] - 1.5,
                f"{pname.split('_')[0]} C", fontsize=6, color=color,
                fontweight="bold", ha="center")

    ax.set_xlabel("X (Å)", fontsize=7)
    ax.set_ylabel("Y (Å)", fontsize=7)
    ax.set_zlabel("Z (Å)", fontsize=7)
    ax.set_title("Predicted Peptide Structures — Cα Trace Overlay", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")

    # Sequence annotation
    seq_text = "Sequences:\n"
    for pname in PEPTIDE_ORDER:
        seq = SEQUENCES.get(pname, "")
        seq_text += f"  {pname.split('_')[0]}: {seq}\n"
    # Note P1/P2/P4 share a sequence
    seq_text += "\nNote: P1, P2, P4 share identical sequence.\nP3 is longer (+Cys C-term)."

    ax.text2D(0.02, 0.02, seq_text, transform=ax.transAxes,
              fontsize=5.5, color="0.3", va="bottom", ha="left",
              fontfamily="monospace",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    footnote = (
        "Structures predicted by ESMFold. Aligned by N-terminal Cα position.\n"
        "P1/P2/P4 share TFFYGGSRGKRNNFKTEGWRGGRL; P3 extends with CVPLPHLKFC.\n"
        "Structural differences arise from predicted folding, not sequence variation."
    )
    fig8.text(0.5, 0.01, footnote, fontsize=6, color="0.4",
              ha="center", fontstyle="italic", transform=fig8.transFigure)
    fig8.subplots_adjust(bottom=0.10)
    fig8.savefig(os.path.join(FIGURES, "Fig8_peptide_overlay.png"))
    plt.close(fig8)
    print(f"    Done: {len(aligned)} peptides aligned")

# ===========================================================================
# FIGURE 9: INTERFACE CONTACT BREAKDOWN
# ===========================================================================
def fig9_contact_breakdown():
    print("  Fig9: Interface contact breakdown...")

    # Sort candidates by total contacts descending
    sorted_cand = sorted(PEPTIDE_ORDER, key=lambda p: -ntof(stats_map[p]["contacts"]))
    all_bars = sorted_cand + ["P5_Scrambled"]

    cand_data = []
    control_data = None
    for pname in all_bars:
        d = stats_map.get(pname)
        if not d:
            continue
        hb = ntof(d["hbonds"])
        hy = ntof(d["hydrophobic"])
        sb = ntof(d["salt_bridges"])
        total = hb + hy + sb
        if pname == "P5_Scrambled":
            control_data = {"hb": hb, "hy": hy, "sb": sb, "total": total}
        else:
            cand_data.append({"name": pname, "hb": hb, "hy": hy, "sb": sb, "total": total})

    fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(8, 4.5))

    # Panel A: Stacked bars
    names = [d["name"] for d in cand_data]
    if control_data:
        names = names + ["Scrambled\n(primary)"]
    x = np.arange(len(names))
    w = 0.5

    hb_vals = [d["hb"] for d in cand_data]
    hy_vals = [d["hy"] for d in cand_data]
    sb_vals = [d["sb"] for d in cand_data]
    if control_data:
        hb_vals.append(control_data["hb"])
        hy_vals.append(control_data["hy"])
        sb_vals.append(control_data["sb"])

    bars_hb = ax9a.bar(x, hb_vals, w, label="H-bonds", color="#43A047",
                       edgecolor="black", linewidth=0.5)
    bars_hy = ax9a.bar(x, hy_vals, w, bottom=hb_vals, label="Hydrophobic",
                       color="#FB8C00", edgecolor="black", linewidth=0.5)
    bottoms_sb = [hb_vals[i] + hy_vals[i] for i in range(len(hb_vals))]
    bars_sb = ax9a.bar(x, sb_vals, w, bottom=bottoms_sb, label="Salt bridges",
                       color="#E53935", edgecolor="black", linewidth=0.5)

    # Total labels
    for i in range(len(names)):
        total = hb_vals[i] + hy_vals[i] + sb_vals[i]
        ax9a.text(i, total + 1.5, f"{int(total)}",
                  ha="center", fontsize=8, fontweight="bold")

    ax9a.set_xticks(x)
    ax9a.set_xticklabels(names, rotation=15, ha="right", fontsize=7.5)
    ax9a.set_ylabel("Contact count", fontsize=9)
    ax9a.set_title("Interface Contact Breakdown by Type", fontsize=9, fontweight="bold")
    ax9a.legend(fontsize=7)

    # Panel B: Proportion (normalized stacked bar)
    for i in range(len(names)):
        total = hb_vals[i] + hy_vals[i] + sb_vals[i]
        if total > 0:
            hb_pct = hb_vals[i] / total * 100
            hy_pct = hy_vals[i] / total * 100
            sb_pct = sb_vals[i] / total * 100
        else:
            hb_pct = hy_pct = sb_pct = 0
        ax9b.bar(i, hb_pct, w, color="#43A047", edgecolor="black", linewidth=0.5)
        ax9b.bar(i, hy_pct, w, bottom=hb_pct, color="#FB8C00",
                 edgecolor="black", linewidth=0.5)
        ax9b.bar(i, sb_pct, w, bottom=hb_pct + hy_pct, color="#E53935",
                 edgecolor="black", linewidth=0.5)
        # Percentage label for hydrophobic (usually dominant)
        if hy_pct > 20:
            ax9b.text(i, hb_pct + hy_pct / 2, f"{hy_pct:.0f}%",
                      ha="center", fontsize=7, fontweight="bold", color="white")
        if hb_pct > 15:
            ax9b.text(i, hb_pct / 2, f"{hb_pct:.0f}%",
                      ha="center", fontsize=7, fontweight="bold", color="white")

    ax9b.set_xticks(x)
    ax9b.set_xticklabels(names, rotation=15, ha="right", fontsize=7.5)
    ax9b.set_ylabel("Proportion (%)", fontsize=9)
    ax9b.set_title("Contact Type Proportion", fontsize=9, fontweight="bold")
    ax9b.set_ylim(0, 110)

    footnote = (
        "Contact types defined by Cα-Cα distance thresholds: H-bonds (3.0–3.9 A),\n"
        "hydrophobic (3.5–5.0 A), salt bridges (<4.0 A with opposite charges).\n"
        "Primary control: P5_Scrambled (n=1, ESMFold). Ensemble control not shown (all 0 contacts)."
    )
    fig9.text(0.5, 0.01, footnote, fontsize=6, color="0.4",
              ha="center", fontstyle="italic", transform=fig9.transFigure)
    fig9.subplots_adjust(bottom=0.15)
    fig9.savefig(os.path.join(FIGURES, "Fig9_contact_breakdown.png"))
    plt.close(fig9)
    print("    Done")

# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    stats_map = {d["peptide"]: d for d in stats}

    print("Generating extended figures...")
    fig6_docking_pose()
    fig7_bbb_radar()
    fig8_peptide_overlay()
    fig9_contact_breakdown()
    print("\nDone. Files in", FIGURES)
    for f in sorted(os.listdir(FIGURES)):
        print(f"  {f}")
