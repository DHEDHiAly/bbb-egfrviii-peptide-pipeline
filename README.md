# BBB-Penetrant EGFRvIII-Targeting Peptide Pre-Validation Ranking Pipeline

Computational prioritization pipeline for BBB-penetrant EGFRvIII-targeting peptides (glioblastoma). Produces ordinal InterfacePriorityScore (IPS) and robustness calibration — without claiming binding affinity, dissociation constants, free energies, or experimental BBB permeability.

## Scope
- **What this pipeline does**: Heuristic ranking of candidate peptides by interface contact count, contact density, IPS composite score, and BBB heuristic criteria.
- **What this pipeline does NOT do**: Predict binding affinity, dissociation constants (Kd), binding free energies (ΔG), experimental BBB permeability, or biological efficacy.
- **Scoring models are heuristic (0.20 per IPS component)** and exploratory. No experimental calibration was performed.

## Control Definitions (critical)
Two distinct control groups — never merged statistically:

| Control type | Source | N | Contacts | Label in figures |
|---|---|---|---|---|
| **Primary control** | P5_Scrambled (ESMFold structure) | 1 | 30 | Scrambled (primary control) |
| **Ensemble control** | 5 calibration scrambles (extended backbone) | 5 | [0,0,0,0,0] | Scrambled ensemble (n=5) |

- Primary control = baseline screening comparator (ranking context).
- Ensemble control = robustness null model (variance/calibration context).
- The ensemble distribution is **degenerate under current CA-contact threshold** (all 0 contacts, zero variance). Cohen's d and raw Z-scores are undefined for this baseline.

## Key Limitations
- **CA-contact model**: Interface contacts computed from Cα distance thresholding. Does not capture side-chain orientation, electrostatics, solvation, or entropy.
- **No ΔG or docking affinity interpretation**: Contact counts are NOT proxies for binding affinity. The docking scoring function is heuristic (clash penalty + distance reward), not physics-based.
- **No experimental validation**: BBB heuristic criteria are four simple physicochemical filters (molecular weight, HBD, HBA, LogP). Not validated against experimental permeability data.
- **Scrambled control n=1 in primary pipeline**: Primary control is a single sequence. No standard-deviation-based statistics computable from primary alone.

## Manuscript Inputs Statement
**Only `/figures` and `/results_final` constitute manuscript inputs. All other directories are archived analytical outputs retained for reproducibility.**

## Repository Structure
```
.
├── pipeline.py                 # Primary end-to-end pipeline (ESMFold structures, docking, scoring)
├── calibration.py              # Robustness analysis (extended backbone, 5 scrambles, 3 scoring models, sensitivity)
├── final_analysis.py           # Statistical lock, methods definitions, claim boundaries, figure datasets
├── generate_figures.py         # Final figure generation (5 manuscript figures from figure_datasets/)
├── figure_provenance.csv       # Figure ID, source dataset, control type, pipeline stage, inclusion status
├── results_final/              # MANUSCRIPT DATASETS (locked statistical outputs, claim boundaries, verdict)
│   └── figure_datasets/        # CSV datasets for figure generation
├── figures/                    # MANUSCRIPT FIGURES (5 files — only ones used in manuscript)
├── archive_full_analysis/      # Archived non-manuscript outputs (see below)
└── bbb_penetrant.py            # Original Colab notebook (reference only)
```

## Figure Provenance
Only final calibrated figures are used in manuscript:

| Figure | Description |
|---|---|
| Fig1_final_consensus_ranking | Bar chart — P3 vs P1/P2/P4 vs primary scrambled control |
| Fig2_control_ensemble_distribution | Two-panel — ensemble control scatter + histogram with degeneracy annotation |
| Fig3_effect_size_heatmap | Capped Z-score grid with class labels (strong/moderate/weak) |
| Fig4_IPS_model_agreement | 2x2 panels — 3 IPS variants, rank consistency table, primary rank, calibration sensitivity |
| Fig5_rank_stability | Two-panel — 125-threshold perturbation stability, primary vs calibration rank comparison |
| Fig6_docking_pose | 3D: receptor Cα trace + P3_Cyclized docked with contact residues highlighted |
| Fig7_BBB_radar | Radar plot — 4 BBB heuristic criteria per peptide (charge, MW, GRAVY, instability) |
| Fig8_peptide_overlay | 3D overlay of predicted Cα traces for all 4 candidate peptides |
| Fig9_contact_breakdown | Stacked bar: HBonds vs hydrophobic vs salt bridges per peptide + proportion panel |

All other figures (7 from pipeline.py, 4 from calibration.py) are archived in `archive_full_analysis/` and not referenced in manuscript.

## Output Requirements
- All figures: matplotlib only, 300 DPI minimum
- No external web servers or manual steps required to reproduce
- All code runs end-to-end locally
