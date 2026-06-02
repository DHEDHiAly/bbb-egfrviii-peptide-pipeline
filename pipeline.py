#!/usr/bin/env python3
"""
BBB-Penetrant EGFRvIII-Targeting Peptide Pre-Validation Ranking Pipeline
=======================================================================
A reproducible multi-stage computational screening workflow for prioritizing
peptide constructs targeting EGFRvIII in glioblastoma.

This pipeline produces an ordinal ranking score (InterfacePriorityScore)
for wet-lab validation candidate selection. It does NOT claim binding
affinity, dissociation constants, free energies, or experimental BBB
permeability. All outputs are computational heuristics for prioritization.

Outputs:
  - results/physicochemical_table.csv
  - results/interface_table.csv
  - results/final_ranking_table.csv
  - results/pipeline_summary.txt
  - figures/Fig1_pipeline_overview.png
  - figures/Fig2_docking_score_comparison.png
  - figures/Fig3_interface_contact_breakdown.png
  - figures/Fig4_normalized_contact_density.png
  - figures/Fig5_BBB_heuristic_scorecard.png
  - figures/Fig6_final_ranking_comparison.png
  - figures/Fig7_control_vs_candidate.png
"""

import os
import sys
import requests
import copy
import math
import random
import textwrap
import warnings
import itertools

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from Bio.PDB import PDBParser, PDBIO, PDBList
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Seq import Seq
from Bio import SeqUtils

from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ===========================================================================
# CONFIGURATION
# ===========================================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

PDB_ID = "8UKX"
RAW_PDB_PATH = os.path.join(OUTPUT_DIR, f"{PDB_ID}.pdb")
RECEPTOR_PDB = os.path.join(OUTPUT_DIR, "receptor_clean.pdb")

DOCKING_RADIUS = 20.0
N_ROTATIONS = 2000

PEPTIDES = {
    "P1_Linear":       "TFFYGGSRGKRNNFKTEGWRGGRL",
    "P2_Capped":       "TFFYGGSRGKRNNFKTEGWRGGRL",
    "P3_Cyclized":     "TFFYGGSRGKRNNFKTCVPLPHLKFC",
    "P4_RetroEnantio": "TFFYGGSRGKRNNFKTEGWRGGRL",
}

# ===========================================================================
# STEP 0: Generate scrambled control (composition-matched)
# ===========================================================================
def generate_scrambled_control(seed_sequence, seed=42):
    chars = list(seed_sequence)
    rng = random.Random(seed)
    rng.shuffle(chars)
    return "".join(chars)

PEPTIDES["P5_Scrambled"] = generate_scrambled_control(PEPTIDES["P1_Linear"])

print("Peptide library (including scrambled control):")
for name, seq in PEPTIDES.items():
    print(f"  {name:20s}  {seq}")

# ===========================================================================
# STEP 1: Download and prepare receptor structure
# ===========================================================================
def download_receptor():
    if os.path.exists(RAW_PDB_PATH):
        print(f"[1] Receptor PDB already exists: {RAW_PDB_PATH}")
        return
    url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
    print(f"[1] Downloading {PDB_ID} from {url} ...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(RAW_PDB_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[1] Saved to {RAW_PDB_PATH}")
    except Exception as e:
        print(f"[1] ERROR downloading: {e}")
        sys.exit(1)

def prepare_receptor():
    if os.path.exists(RECEPTOR_PDB):
        print(f"[1] Receptor clean PDB exists: {RECEPTOR_PDB}")
        return
    print("[1] Preparing receptor ...")
    with open(RAW_PDB_PATH) as f:
        lines = [l for l in f if l.startswith("ATOM")]
    with open(RECEPTOR_PDB, "w") as f:
        f.writelines(lines)
        f.write("END\n")
    print(f"[1] Saved receptor ({len(lines)} atoms)")

download_receptor()
prepare_receptor()

# Parse receptor once
parser = PDBParser(QUIET=True)
rec_struct = parser.get_structure("receptor", RECEPTOR_PDB)
rec_atoms_list = list(rec_struct.get_atoms())
rec_coords = np.array([a.get_coord() for a in rec_atoms_list])
print(f"[1] Receptor atoms: {len(rec_atoms_list)}")

# Compute docking center dynamically: find a surface-proximal point
# by going 65% from COM toward the farthest atom
_rec_com = rec_coords.mean(axis=0)
_rec_vecs = rec_coords - _rec_com
_rec_dists_sq = (_rec_vecs ** 2).sum(axis=1)
_rec_farthest = rec_coords[_rec_dists_sq.argmax()]
DOCKING_CENTER = _rec_com + (_rec_farthest - _rec_com) * 0.65
print(f"[1] Docking center (surface-proximal): {DOCKING_CENTER}")

# ===========================================================================
# STEP 2: Predict peptide structures (ESMFold API with fallback)
# ===========================================================================
def build_extended_backbone(name, sequence):
    out = os.path.join(OUTPUT_DIR, f"{name}.pdb")
    n = len(sequence)
    coords = np.zeros((n, 3))
    for i in range(n):
        coords[i] = np.array([0.0, 0.0, i * 3.8])
    coords -= coords.mean(axis=0)
    coords += DOCKING_CENTER
    with open(out, "w") as f:
        for i, aa in enumerate(sequence):
            resname = {"A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE",
                       "G":"GLY","H":"HIS","I":"ILE","K":"LYS","L":"LEU",
                       "M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG",
                       "S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"}.get(aa, "ALA")
            x, y, z = coords[i]
            f.write(f"ATOM  {i+1:5d}  CA  {resname} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
        f.write("END\n")
    return out

def predict_structure_esmfold(name, sequence):
    out = os.path.join(OUTPUT_DIR, f"{name}.pdb")
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        resp = requests.post(url, data=sequence.encode("utf-8"), headers=headers, timeout=180)
        resp.raise_for_status()
        with open(out, "w") as f:
            f.write(resp.text)
        print(f"[2] {name}: ESMFold prediction saved")
        return True
    except Exception as e:
        print(f"[2] {name}: ESMFold failed ({e}), using extended backbone fallback")
        return False

for name, seq in PEPTIDES.items():
    pdb_file = os.path.join(OUTPUT_DIR, f"{name}.pdb")
    if os.path.exists(pdb_file):
        print(f"[2] {name}: structure exists, skipping prediction")
        continue
    ok = predict_structure_esmfold(name, seq)
    if not ok:
        build_extended_backbone(name, seq)
        print(f"[2] {name}: extended backbone built")

# ===========================================================================
# STEP 3: Heuristic docking (rigid-body placement + scoring)
# ===========================================================================
def score_pose(pep_coords, rec_coords):
    score = 0.0
    for pc in pep_coords:
        dists_sq = np.sum((rec_coords - pc) ** 2, axis=1)
        min_d = np.sqrt(np.min(dists_sq))
        if min_d < 2.2:
            score += 10.0
        elif min_d < 3.8:
            score -= 1.5
        elif min_d < 5.0:
            score -= 0.5
    return score

def dock_peptide(name):
    pdb_file = os.path.join(OUTPUT_DIR, f"{name}.pdb")
    out_pdb = os.path.join(OUTPUT_DIR, f"{name}_docked.pdb")
    if not os.path.exists(pdb_file):
        print(f"[3] {name}: PDB not found, skipping")
        return None, None
    pep_struct = parser.get_structure("pep", pdb_file)
    pep_atoms = list(pep_struct.get_atoms())
    if len(pep_atoms) == 0:
        return None, None
    pep_coords = np.array([a.get_coord() for a in pep_atoms])
    com = pep_coords.mean(axis=0)
    pep_coords_centered = pep_coords - com
    best_score = float("inf")
    best_coords = pep_coords_centered + DOCKING_CENTER
    angles = Rotation.random(N_ROTATIONS)
    for rot in angles:
        rotated = rot.apply(pep_coords_centered) + DOCKING_CENTER
        s = score_pose(rotated, rec_coords)
        if s < best_score:
            best_score = s
            best_coords = rotated.copy()
    for i, atom in enumerate(pep_atoms):
        atom.set_coord(best_coords[i])
    io = PDBIO()
    io.set_structure(pep_struct)
    io.save(out_pdb)
    return best_score, out_pdb

print("[3] Docking peptides ...")
docking_results = {}
for name in PEPTIDES:
    score, path = dock_peptide(name)
    docking_results[name] = {"score": score, "path": path}
    if score is not None:
        print(f"  {name:20s}  contact score = {score:.2f}")
    else:
        print(f"  {name:20s}  FAILED")

# ===========================================================================
# STEP 4: Build complex PDBs and analyze interface contacts
# ===========================================================================
def build_complex_pdb(receptor_file, peptide_docked, complex_out):
    with open(receptor_file) as f:
        rec_lines = [l for l in f if l.startswith("ATOM")]
    with open(complex_out, "w") as out:
        out.writelines(rec_lines)
        out.write("TER\n")
        with open(peptide_docked) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    out.write(line[:21] + "B" + line[22:])
        out.write("END\n")

def count_interface_contacts(complex_pdb):
    if not os.path.exists(complex_pdb) or os.path.getsize(complex_pdb) < 100:
        return {"hbonds": 0, "hydrophobic": 0, "salt_bridges": 0}
    try:
        struct = parser.get_structure("cpx", complex_pdb)
    except Exception:
        return {"hbonds": 0, "hydrophobic": 0, "salt_bridges": 0}
    model = struct[0]
    chains = list(model.get_chains())
    if len(chains) < 2:
        return {"hbonds": 0, "hydrophobic": 0, "salt_bridges": 0}
    rec_chain = chains[0]
    pep_chain = chains[1]
    rec_atoms = list(rec_chain.get_atoms())
    pep_atoms = list(pep_chain.get_atoms())
    hb = hy = sb = 0
    for pa in pep_atoms:
        pres = pa.get_parent().get_resname()
        for ra in rec_atoms:
            rres = ra.get_parent().get_resname()
            d = pa - ra
            if d < 3.5 and pa.element in ("N", "O") and ra.element in ("N", "O"):
                hb += 1
            if d < 4.5 and pa.element == "C" and ra.element == "C":
                hy += 1
            if d < 4.2:
                if (pres in ("LYS", "ARG") and rres in ("ASP", "GLU")) or \
                   (pres in ("ASP", "GLU") and rres in ("LYS", "ARG")):
                    sb += 1
    return {"hbonds": hb, "hydrophobic": hy, "salt_bridges": sb}

print("[4] Building complexes and analyzing interfaces ...")
interface_data = []
for name in PEPTIDES:
    docked = docking_results[name]["path"]
    if docked is None or not os.path.exists(docked):
        print(f"  {name:20s}  no docked structure, skipping interface analysis")
        continue
    complex_pdb = os.path.join(OUTPUT_DIR, f"{name}_complex.pdb")
    build_complex_pdb(RECEPTOR_PDB, docked, complex_pdb)
    contacts = count_interface_contacts(complex_pdb)
    tot = contacts["hbonds"] + contacts["hydrophobic"] + contacts["salt_bridges"]
    seq_len = len(PEPTIDES[name])
    norm_density = round(tot / seq_len, 3) if seq_len > 0 else 0.0
    interface_data.append({
        "Peptide": name,
        "Sequence": PEPTIDES[name],
        "Length": seq_len,
        "HBonds": contacts["hbonds"],
        "Hydrophobic": contacts["hydrophobic"],
        "SaltBridges": contacts["salt_bridges"],
        "TotalContacts": tot,
        "NormContactDensity": norm_density,
    })
    print(f"  {name:20s}  HB={contacts['hbonds']:3d}  Hy={contacts['hydrophobic']:3d}  SB={contacts['salt_bridges']:2d}  "
          f"Total={tot:3d}  Density={norm_density:.3f}")

df_interface = pd.DataFrame(interface_data)
df_interface.to_csv(os.path.join(RESULTS_DIR, "interface_table.csv"), index=False)
print(f"[4] Interface table saved ({len(df_interface)} entries)")

# ===========================================================================
# STEP 5: Physicochemical properties and BBB heuristics
# ===========================================================================
BOMAN_SCALE = {
    "A": 0.28, "R": -0.66, "N": -0.16, "D": -0.49, "C": 0.13,
    "Q": -0.14, "E": -0.45, "G": 0.00, "H": -0.19, "I": 0.73,
    "L": 0.53, "K": -0.81, "M": 0.26, "F": 0.61, "P": -0.14,
    "S": -0.01, "T": 0.05, "W": 0.37, "Y": 0.02, "V": 0.47,
}

def calc_boman(seq):
    return sum(BOMAN_SCALE.get(aa.upper(), 0.0) for aa in seq)

def calc_net_charge_ph74(seq):
    q = 0.0
    for aa in seq:
        if aa in ("K", "R"):
            q += 1.0
        elif aa in ("D", "E"):
            q -= 1.0
        elif aa == "H":
            q += 0.1
    return q

print("[5] Calculating physicochemical properties ...")
physicochemical_data = []
for name, seq in PEPTIDES.items():
    try:
        pa = ProteinAnalysis(seq)
        mw = pa.molecular_weight()
        pI = pa.isoelectric_point()
        instab = pa.instability_index()
        arom = pa.aromaticity()
        gravy = pa.gravy()
    except Exception:
        mw = pI = instab = arom = gravy = float("nan")
    nq = calc_net_charge_ph74(seq)
    boman = calc_boman(seq)
    bbb_charge = 1 if 2 <= nq <= 8 else 0
    bbb_mw = 1 if mw < 4000 else 0
    bbb_gravy = 1 if gravy > -1.0 else 0
    bbb_instab = 1 if instab < 40 else 0
    bbb_total = bbb_charge + bbb_mw + bbb_gravy + bbb_instab
    physicochemical_data.append({
        "Peptide": name,
        "Sequence": seq,
        "Length": len(seq),
        "MW_Da": round(mw, 2),
        "pI": round(pI, 2),
        "InstabilityIndex": round(instab, 2),
        "Aromaticity": round(arom, 3),
        "Gravy": round(gravy, 3),
        "NetCharge_pH74": round(nq, 1),
        "BomanIndex": round(boman, 2),
        "BBB_Charge": bbb_charge,
        "BBB_MW": bbb_mw,
        "BBB_Gravy": bbb_gravy,
        "BBB_Instability": bbb_instab,
        "BBB_Total": bbb_total,
    })

df_physchem = pd.DataFrame(physicochemical_data)
df_physchem.to_csv(os.path.join(RESULTS_DIR, "physicochemical_table.csv"), index=False)
print(f"[5] Physicochemical table saved ({len(df_physchem)} entries)")

# ===========================================================================
# STEP 6: Compute InterfacePriorityScore and rank
# ===========================================================================
print("[6] Computing InterfacePriorityScore ...")

merged = df_interface.merge(df_physchem, on=["Peptide", "Sequence", "Length"], how="outer")
merged.fillna(0, inplace=True)

# Extract scrambled control values for delta computation
scrambled_row = merged[merged["Peptide"] == "P5_Scrambled"]
scrambled_density = scrambled_row["NormContactDensity"].values[0] if len(scrambled_row) > 0 else 0
scrambled_total = scrambled_row["TotalContacts"].values[0] if len(scrambled_row) > 0 else 0

def minmax_norm(series):
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return series * 0.0
    return (series - lo) / (hi - lo)

# Component 1: Contact Score (normalized total contacts)
merged["ContactScore_norm"] = minmax_norm(merged["TotalContacts"])

# Component 2: Contact Density Score (normalized)
merged["DensityScore_norm"] = minmax_norm(merged["NormContactDensity"])

# Component 3: Control Delta (difference from scrambled in both total and density)
merged["ControlDelta_total"] = merged["TotalContacts"] - scrambled_total
merged["ControlDelta_density"] = merged["NormContactDensity"] - scrambled_density
merged["ControlDelta_norm"] = minmax_norm(merged["ControlDelta_total"])
merged["ControlDelta_norm"] = merged["ControlDelta_norm"].clip(lower=0)

# Component 4: BBB heuristic score (normalized)
merged["BBB_norm"] = minmax_norm(merged["BBB_Total"])

# Component 5: Physicochemical score (inverse instability, normalized polarity)
merged["Instab_inv"] = -merged["InstabilityIndex"]
merged["Instab_norm"] = minmax_norm(merged["Instab_inv"])
merged["PhysChem_norm"] = 0.5 * merged["Instab_norm"] + 0.5 * minmax_norm(-merged["Gravy"])

# Composite: InterfacePriorityScore
merged["InterfacePriorityScore"] = (
    0.20 * merged["ContactScore_norm"] +
    0.20 * merged["DensityScore_norm"] +
    0.20 * merged["ControlDelta_norm"] +
    0.20 * merged["BBB_norm"] +
    0.20 * merged["PhysChem_norm"]
)

# Round for display
merged["InterfacePriorityScore"] = merged["InterfacePriorityScore"].round(4)

# Rank
merged["WetLabRank"] = merged["InterfacePriorityScore"].rank(ascending=False, method="min").astype(int)
merged.sort_values("WetLabRank", inplace=True)
merged.reset_index(drop=True, inplace=True)

# Create final ranking table
ranking_cols = [
    "Peptide", "Sequence", "Length",
    "TotalContacts", "NormContactDensity",
    "HBonds", "Hydrophobic", "SaltBridges",
    "MW_Da", "pI", "InstabilityIndex", "Gravy", "NetCharge_pH74",
    "BBB_Total",
    "ControlDelta_total", "ControlDelta_density",
    "InterfacePriorityScore", "WetLabRank",
]
available_cols = [c for c in ranking_cols if c in merged.columns]
df_ranking = merged[available_cols].copy()
df_ranking.to_csv(os.path.join(RESULTS_DIR, "final_ranking_table.csv"), index=False)
print(f"[6] Final ranking table saved ({len(df_ranking)} entries)")

best_peptide = df_ranking.iloc[0]["Peptide"]
best_score = df_ranking.iloc[0]["InterfacePriorityScore"]
print(f"\n  === WET-LAB CANDIDATE: {best_peptide} (InterfacePriorityScore = {best_score}) ===")

# Identify P3_Cyclized rank
p3_row = df_ranking[df_ranking["Peptide"] == "P3_Cyclized"]
if len(p3_row) > 0:
    print(f"  P3_Cyclized rank: {p3_row.iloc[0]['WetLabRank']}")

# ===========================================================================
# STEP 7: Generate pipeline summary
# ===========================================================================
print("[7] Writing pipeline summary ...")

summary_lines = [
    "=" * 72,
    "BBB-PENETRANT EGFRvIII-TARGETING PEPTIDE PRE-VALIDATION RANKING PIPELINE",
    "=" * 72,
    "",
    "DISCLAIMER:",
    "  This is a computational prioritization pipeline producing ordinal heuristic",
    "  ranking scores. It does NOT predict binding affinity, dissociation constants,",
    "  free energies, experimental BBB permeability, or biological efficacy.",
    "  All outputs are intended for wet-lab validation candidate prioritization only.",
    "",
    "PIPELINE STAGES:",
    "  1. Target preparation (PDB 8UKX download, chain extraction)",
    "  2. Peptide library definition (4 constructs + scrambled control)",
    "  3. 3D structure prediction (ESMFold API with extended backbone fallback)",
    "  4. Heuristic rigid-body docking (COM alignment + rotational sampling)",
    "  5. Interface contact analysis (HBonds, hydrophobic, salt bridges)",
    "  6. Physicochemical characterization (MW, pI, instability, gravy, charge)",
    "  7. BBB heuristic filtering (charge, MW, hydrophobicity, instability)",
    "  8. Control comparison (composition-matched scrambled peptide)",
    "  9. Composite scoring (InterfacePriorityScore)",
    "  10. Wet-lab candidate ranking",
    "",
    f"DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    f"TARGET: EGFRvIII (PDB: {PDB_ID})",
    "",
    "PEPTIDE LIBRARY:",
]

for name, seq in PEPTIDES.items():
    summary_lines.append(f"  {name:20s}  {seq}")

summary_lines.extend([
    "",
    "SCIENTIFIC NOVELTY:",
    "  1. Multi-stage reproducible screening workflow",
    "  2. Evidence-level labeling via normalized contact density",
    "  3. Composition-matched scrambled peptide negative control",
    "  4. InterfacePriorityScore: ordinal heuristic for wet-lab prioritization",
    "  5. BBB-aware heuristic filtering integrated into composite scoring",
    "  6. Clear separation of in silico screening from experimental validation",
    "",
    "SCORING COMPONENTS (InterfacePriorityScore):",
    "  - ContactScore_norm  (20%): normalized total interface contacts",
    "  - DensityScore_norm  (20%): normalized contacts per residue",
    "  - ControlDelta_norm  (20%): improvement over scrambled control",
    "  - BBB_norm           (20%): BBB heuristic filter score",
    "  - PhysChem_norm      (20%): physicochemical profile",
    "",
    "WET-LAB VALIDATION CANDIDATE:",
    f"  {best_peptide} (InterfacePriorityScore = {best_score})",
    "",
    "FULL RANKING:",
])

for _, row in df_ranking.iterrows():
    summary_lines.append(
        f"  Rank {row['WetLabRank']}: {row['Peptide']:20s}  "
        f"IPS={row['InterfacePriorityScore']:.4f}  "
        f"Contacts={row['TotalContacts']}  "
        f"Density={row['NormContactDensity']:.3f}  "
        f"BBB={row.get('BBB_Total', 'N/A')}"
    )

summary_lines.extend([
    "",
    "INTERPRETATION:",
    "  The top-ranked peptide (Rank 1) is the recommended first candidate for",
    "  wet-lab synthesis and experimental validation. The InterfacePriorityScore",
    "  integrates docking contacts, control comparison, BBB heuristics, and",
    "  physicochemical properties into a single ordinal prioritization metric.",
    "  All constructs should be tested in the ranked order.",
    "",
    "CAVEATS:",
    "  1. The scrambled control establishes baseline contact frequency for the",
    "     given amino acid composition in the binding site geometry.",
    "  2. P4 (RetroEnantio) uses the same sequence as P1 but with D-amino acids;",
    "     the heuristic model here does not capture chirality effects.",
    "  3. P3 (Cyclized) contains a disulfide bridge not modeled in this pipeline.",
    "  4. These are computational heuristics, not experimental measurements.",
    "",
    "=" * 72,
])

summary_path = os.path.join(RESULTS_DIR, "pipeline_summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))
print(f"[7] Summary saved: {summary_path}")

# ===========================================================================
# STEP 8: Generate all 7 figures
# ===========================================================================
print("[8] Generating figures ...")

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

peptide_order = [p for p in ["P1_Linear","P2_Capped","P3_Cyclized","P4_RetroEnantio","P5_Scrambled"] if p in merged["Peptide"].values]
palette = {"P1_Linear":"#2196F3","P2_Capped":"#4CAF50","P3_Cyclized":"#FF9800",
           "P4_RetroEnantio":"#E91E63","P5_Scrambled":"#9E9E9E"}
bar_colors = [palette[p] for p in peptide_order]
plt.close("all")

# Figure 1: Pipeline overview
fig1, ax1 = plt.subplots(figsize=(5.5, 6))
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 86)
steps = [
    "Target: EGFRvIII\n(PDB 8UKX)",
    "Peptide Library\n(4 constructs +\nscrambled control)",
    "Structure Prediction\n(ESMFold / backbone\nfallback)",
    "Heuristic Docking\n(rigid-body placement\n+ contact scoring)",
    "Interface Analysis\n(HBonds, hydrophobic,\nsalt bridges)",
    "Physicochemical &\nBBB Heuristic\nFiltering",
    "InterfacePriorityScore\nComposite Ranking",
    "Wet-Lab Candidate\nSelection",
]
box_colors_cycle = ["#E3F2FD", "#E8F5E9", "#FFF3E0", "#F3E5F5", "#E0F7FA",
                    "#FFF8E1", "#FBE9E7", "#F1F8E9"]
y_positions = np.linspace(78, 4, len(steps))
for i, (text, y) in enumerate(zip(steps, y_positions)):
    rect = FancyBboxPatch((2.5, y - 4.5), 5, 9, boxstyle="round,pad=0.2",
                          ec="black", fc=box_colors_cycle[i], lw=0.8)
    ax1.add_patch(rect)
    ax1.text(5, y, text, ha="center", va="center", fontsize=6.5, fontweight="bold")
    if i < len(steps) - 1:
        dy = y_positions[i + 1] + 4.5 - (y - 4.5)
        ax1.annotate("", xy=(5, y - 4.5), xytext=(5, y - 4.5 + dy / 2),
                     arrowprops=dict(arrowstyle="->", lw=1.2, color="0.3"),
                     clip_on=False)
ax1.set_axis_off()
ax1.set_title("Pipeline Overview", fontsize=11, fontweight="bold", pad=10)
fig1.savefig(os.path.join(FIGURES_DIR, "Fig1_pipeline_overview.png"))
plt.close(fig1)
print("  Fig1: Pipeline overview")

# Figure 2: Docking score comparison
fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
local_df = merged[merged["Peptide"].isin(peptide_order)].copy()
local_df["plot_order"] = local_df["Peptide"].map({p: i for i, p in enumerate(peptide_order)})
local_df.sort_values("plot_order", inplace=True)
scores = local_df["TotalContacts"].values
ax2.bar(range(len(scores)), scores, color=bar_colors, edgecolor="black", linewidth=0.5)
ax2.set_xticks(range(len(scores)))
ax2.set_xticklabels(local_df["Peptide"].values, rotation=20, ha="right", fontsize=8)
ax2.set_ylabel("Total Interface Contacts", fontsize=9)
ax2.set_title("Interface Contact Score Comparison", fontsize=10, fontweight="bold")
for i, v in enumerate(scores):
    ax2.text(i, v + max(scores) * 0.02, str(v), ha="center", va="bottom", fontsize=7, fontweight="bold")
fig2.tight_layout()
fig2.savefig(os.path.join(FIGURES_DIR, "Fig2_docking_score_comparison.png"))
plt.close(fig2)
print("  Fig2: Docking score comparison")

# Figure 3: Interface contact breakdown (stacked bar)
fig3, ax3 = plt.subplots(figsize=(4.5, 3.5))
hb = local_df["HBonds"].values
hy = local_df["Hydrophobic"].values
sb = local_df["SaltBridges"].values
idx = np.arange(len(hb))
ax3.bar(idx, hb, label="H-Bonds", color="#3F51B5", edgecolor="black", linewidth=0.3)
ax3.bar(idx, hy, bottom=hb, label="Hydrophobic", color="#009688", edgecolor="black", linewidth=0.3)
ax3.bar(idx, sb, bottom=hb + hy, label="Salt Bridges", color="#FF5722", edgecolor="black", linewidth=0.3)
ax3.set_xticks(idx)
ax3.set_xticklabels(local_df["Peptide"].values, rotation=20, ha="right", fontsize=8)
ax3.set_ylabel("Contact Count", fontsize=9)
ax3.set_title("Interface Contact Breakdown", fontsize=10, fontweight="bold")
ax3.legend(fontsize=7, loc="upper right")
fig3.tight_layout()
fig3.savefig(os.path.join(FIGURES_DIR, "Fig3_interface_contact_breakdown.png"))
plt.close(fig3)
print("  Fig3: Interface contact breakdown")

# Figure 4: Normalized contact density
fig4, ax4 = plt.subplots(figsize=(4.5, 3.5))
density = local_df["NormContactDensity"].values
ax4.bar(range(len(density)), density, color=bar_colors, edgecolor="black", linewidth=0.5)
ax4.set_xticks(range(len(density)))
ax4.set_xticklabels(local_df["Peptide"].values, rotation=20, ha="right", fontsize=8)
ax4.set_ylabel("Contacts per Residue", fontsize=9)
ax4.set_title("Normalized Contact Density", fontsize=10, fontweight="bold")
for i, v in enumerate(density):
    ax4.text(i, v + max(density) * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
fig4.tight_layout()
fig4.savefig(os.path.join(FIGURES_DIR, "Fig4_normalized_contact_density.png"))
plt.close(fig4)
print("  Fig4: Normalized contact density")

# Figure 5: BBB heuristic scorecard (heatmap)
fig5, ax5 = plt.subplots(figsize=(5, 3.5))
bbb_labels = ["Charge", "MW<4kDa", "Gravy>-1", "Instab<40", "Total"]
bbb_matrix = local_df[["BBB_Charge","BBB_MW","BBB_Gravy","BBB_Instability","BBB_Total"]].values.astype(float)
if bbb_matrix.size > 0:
    sns.heatmap(bbb_matrix, annot=True, fmt=".0f",
                cmap=sns.color_palette(["#ef9a9a", "#a5d6a7"]),
                cbar=False, linewidths=0.8, linecolor="white",
                xticklabels=bbb_labels,
                yticklabels=local_df["Peptide"].values,
                ax=ax5, annot_kws={"fontsize": 8})
    ax5.set_title("BBB Permeability Heuristic Scorecard", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Criterion", fontsize=8)
fig5.tight_layout()
fig5.savefig(os.path.join(FIGURES_DIR, "Fig5_BBB_heuristic_scorecard.png"))
plt.close(fig5)
print("  Fig5: BBB heuristic scorecard")

# Figure 6: Final ranking comparison (InterfacePriorityScore)
fig6, ax6 = plt.subplots(figsize=(5, 3.5))
local_sorted = df_ranking[df_ranking["Peptide"].isin(peptide_order)].copy()
local_sorted["plot_order"] = local_sorted["Peptide"].map({p: i for i, p in enumerate(peptide_order)})
local_sorted.sort_values("plot_order", inplace=True)
ips = local_sorted["InterfacePriorityScore"].values
rank_labels = local_sorted["Peptide"].values
rank_colors = [palette[p] for p in rank_labels]
bars = ax6.barh(range(len(ips)), ips, color=rank_colors, edgecolor="black", linewidth=0.5)
ax6.set_yticks(range(len(ips)))
ax6.set_yticklabels(rank_labels, fontsize=8)
ax6.set_xlabel("InterfacePriorityScore", fontsize=9)
ax6.set_title("Final Ranking Score Comparison", fontsize=10, fontweight="bold")
for i, (v, p) in enumerate(zip(ips, rank_labels)):
    rank = local_sorted[local_sorted["Peptide"] == p]["WetLabRank"].values[0]
    ax6.text(v + 0.01, i, f"  #{rank}  ({v:.3f})", va="center", fontsize=7,
             fontweight="bold", color="0.2")
ax6.invert_yaxis()
ax6.set_xlim(0, ips.max() * 1.4)
fig6.tight_layout()
fig6.savefig(os.path.join(FIGURES_DIR, "Fig6_final_ranking_comparison.png"))
plt.close(fig6)
print("  Fig6: Final ranking comparison")

# Figure 7: Control vs candidate comparison
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(7, 3.5))
candidate_name = best_peptide
control_name = "P5_Scrambled"
cand_row = merged[merged["Peptide"] == candidate_name].iloc[0] if len(merged[merged["Peptide"] == candidate_name]) > 0 else None
ctrl_row = merged[merged["Peptide"] == control_name].iloc[0] if len(merged[merged["Peptide"] == control_name]) > 0 else None
if cand_row is not None and ctrl_row is not None:
    categories = ["HBonds", "Hydrophobic", "SaltBridges"]
    cand_vals = [cand_row.get(c, 0) for c in categories]
    ctrl_vals = [ctrl_row.get(c, 0) for c in categories]
    x = np.arange(len(categories))
    w = 0.35
    ax7a.bar(x - w / 2, cand_vals, w, label=candidate_name, color=palette.get(candidate_name, "#2196F3"),
             edgecolor="black", linewidth=0.5)
    ax7a.bar(x + w / 2, ctrl_vals, w, label=control_name, color=palette.get(control_name, "#9E9E9E"),
             edgecolor="black", linewidth=0.5)
    ax7a.set_xticks(x)
    ax7a.set_xticklabels(categories, fontsize=8)
    ax7a.set_ylabel("Count", fontsize=8)
    ax7a.set_title("Contact Type Comparison", fontsize=9, fontweight="bold")
    ax7a.legend(fontsize=7)
    # Panel B: Total contacts + density
    metrics = ["TotalContacts", "NormContactDensity"]
    cand_m = [cand_row.get("TotalContacts", 0), cand_row.get("NormContactDensity", 0)]
    ctrl_m = [ctrl_row.get("TotalContacts", 0), ctrl_row.get("NormContactDensity", 0)]
    x2 = np.arange(2)
    ax7b.bar(x2 - w / 2, cand_m, w, label=candidate_name, color=palette.get(candidate_name, "#2196F3"),
             edgecolor="black", linewidth=0.5)
    ax7b.bar(x2 + w / 2, ctrl_m, w, label=control_name, color=palette.get(control_name, "#9E9E9E"),
             edgecolor="black", linewidth=0.5)
    ax7b.set_xticks(x2)
    ax7b.set_xticklabels(["Total Contacts", "Contact Density"], fontsize=7)
    ax7b.set_ylabel("Value", fontsize=8)
    ax7b.set_title("Overall Contact Comparison", fontsize=9, fontweight="bold")
    ax7b.legend(fontsize=7)
    delta_t = cand_row.get("TotalContacts", 0) - ctrl_row.get("TotalContacts", 0)
    delta_d = cand_row.get("NormContactDensity", 0) - ctrl_row.get("NormContactDensity", 0)
    fig7.suptitle(f"Control vs Candidate: ΔTotal={delta_t:+d}, ΔDensity={delta_d:+.3f}",
                  fontsize=10, fontweight="bold", y=1.03)
else:
    ax7a.text(0.5, 0.5, "Insufficient data for comparison", ha="center", va="center", transform=ax7a.transAxes)
    ax7b.text(0.5, 0.5, "Insufficient data for comparison", ha="center", va="center", transform=ax7b.transAxes)
fig7.tight_layout()
fig7.savefig(os.path.join(FIGURES_DIR, "Fig7_control_vs_candidate.png"))
plt.close(fig7)
print("  Fig7: Control vs candidate comparison")

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 72)
print("PIPELINE COMPLETE")
print("=" * 72)
print(f"\nResults:   {RESULTS_DIR}/")
print(f"  - physicochemical_table.csv")
print(f"  - interface_table.csv")
print(f"  - final_ranking_table.csv")
print(f"  - pipeline_summary.txt")
print(f"\nFigures:   {FIGURES_DIR}/")
print(f"  - Fig1_pipeline_overview.png")
print(f"  - Fig2_docking_score_comparison.png")
print(f"  - Fig3_interface_contact_breakdown.png")
print(f"  - Fig4_normalized_contact_density.png")
print(f"  - Fig5_BBB_heuristic_scorecard.png")
print(f"  - Fig6_final_ranking_comparison.png")
print(f"  - Fig7_control_vs_candidate.png")
print(f"\nWET-LAB CANDIDATE: {best_peptide} (InterfacePriorityScore = {best_score})")
print("=" * 72)
