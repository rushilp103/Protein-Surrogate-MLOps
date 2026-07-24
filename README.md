# Protein-Surrogate-ML

XGBoost surrogate model that predicts AutoDock Vina docking scores from AlphaFold-derived structural and biochemical features.

## Environment

All project work must use the `protein` conda environment:

```bash
conda activate protein
pip install -r requirements.txt
```

## Configuration

Pipeline settings live under `configs/`. The EGFR micro-dataset used for early development:

```bash
configs/egfr.yaml   # protein, drug, 3 mutations (L858R, T790M, G719S), paths
```

Load and validate a config:

```python
from src.config import load_config

config = load_config("configs/egfr.yaml")
```

## Step 2: Wild-type preparation

Downloads the AlphaFold EGFR PDB and Gefitinib SDF, cleans heteroatoms/water from the WT structure, and writes a ligand PDBQT:

```bash
conda activate protein
python -m src.download configs/egfr.yaml
```

Outputs:

- `data/raw/AF-P00533-F1.pdb` — AlphaFold structure
- `data/raw/AF-P00533-F1_clean.pdb` — standard amino acids only (mutation-ready)
- `data/raw/Gefitinib.sdf` — 3D ligand (PubChem + RDKit embed if needed)
- `data/docking/Gefitinib.pdbqt` — Vina-ready ligand

SDF→PDBQT prefers OpenBabel (Python API via `openbabel-wheel`, then `obabel` CLI); Meeko/RDKit is the fallback.

## Repository layout

```
├── configs/           # YAML pipeline configs
├── src/               # Offline pipeline + FastAPI inference
├── data/              # raw / mutants / docking / processed
├── models/            # trained artifacts
├── outputs/           # metrics, figures, reports
└── tests/
```

## Tests

```bash
conda activate protein
pytest
```
