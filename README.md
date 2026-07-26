# Protein-Surrogate-ML

XGBoost surrogate model that predicts AutoDock Vina docking scores from AlphaFold-derived structural and biochemical features.

## Environment

All project work **must** use an activated `protein` conda environment (do not run with system Python or a bare `python.exe` path without activation — activation supplies `Library\bin` on `PATH` for native DLLs such as `libiomp5md.dll`):

```bash
conda activate protein
pip install -r requirements.txt
```

## Two ways to use this repo

| Path | What it does | Needs OpenMM / Vina? |
| --- | --- | --- |
| **Offline pipeline** | Mutate → dock → features → train `xgboost_model.pkl` | Yes (local `protein` env) |
| **Inference API** | Serve `/predict` and `/explain` from precomputed artifacts | No (Docker or lightweight conda) |

Generate data and train first; then run the API locally or with Docker.

---

## Configuration

```bash
configs/egfr.yaml   # EGFR + Gefitinib, ~100 kinase-domain point mutations, paths, docking box, training
```

Load and validate:

```python
from src.config import load_config

config = load_config("configs/egfr.yaml")
```

Training keys in the YAML: `target_column`, `group_column` (`residue_position`), `n_splits` (GroupKFold), `test_size` (GroupShuffleSplit hold-out), `random_state`.

---

## Offline pipeline (data generation + training)

Requires the full `protein` environment, AutoDock Vina on `PATH` / `VINA_BINARY` / `tools/vina.exe`, and time proportional to the mutation list (often overnight for the macro set).

### One-shot runner

```bash
conda activate protein
python -m src.pipeline configs/egfr.yaml
```

Useful flags:

- `--skip-download` — reuse existing WT / ligand files under `data/raw`
- `--force` — regenerate mutants and docking even if outputs exist

### Step-by-step

**Step 2 — Wild-type preparation**

```bash
python -m src.download configs/egfr.yaml
```

Outputs: `data/raw/AF-P00533-F1.pdb`, `AF-P00533-F1_clean.pdb`, `Gefitinib.sdf`, `data/docking/Gefitinib.pdbqt`.

**Step 3 — Physics (OpenMM + Vina)**

```bash
python -m src.openmm_mutator configs/egfr.yaml
python -m src.docking configs/egfr.yaml
```

Outputs: `data/mutants/{MUTATION}.pdb`, docking PDBQTs/logs, `data/processed/labels.csv`.

Existing mutant/docking files are skipped unless you pass `--force`.

**Step 4 — Feature extraction**

```bash
python -m src.features configs/egfr.yaml
```

| Column | Description |
| --- | --- |
| `dist_to_ligand` | Cα → ligand centroid distance (Å) |
| `dist_to_pocket` | Cα → docking box center distance (Å) |
| `sasa` | FreeSASA residue SASA (Å²) |
| `delta_charge` | Formal charge change (mut − wt) |
| `delta_volume` | Residue volume change (Å³, mut − wt) |

Output: `data/processed/features.csv`.

**Step 5 / 7 — Train surrogate**

```bash
python -m src.train configs/egfr.yaml
```

Training behavior (macro-dataset):

1. Merge `features.csv` + `labels.csv` on `mutation`
2. Set `mutation` as the row index; drop `wt_aa` / `mut_aa` (matrix stays numeric)
3. **GroupShuffleSplit** 80/20 hold-out by `residue_position` (no residue leakage into test)
4. **GridSearchCV** + **GroupKFold** over `max_depth` ∈ {4,5,6}, `n_estimators` ∈ {100,200,300}, `learning_rate` ∈ {0.01,0.05,0.1}
5. Report hold-out RMSE / MAE / R²; refit final model on **all** rows with `best_params`
6. Write Actual vs Predicted for the hold-out set

Outputs:

- `models/xgboost_model.pkl` — regressor + feature names + `best_params`
- `outputs/metrics/actual_vs_predicted.csv` — hold-out Actual vs Predicted
- `outputs/metrics/training_metrics.json` — CV / hold-out summary
- `outputs/figures/feature_importances.png`

### AutoDock Vina binary

`src.docking` shells out to the Vina CLI via:

1. `PATH` (`vina` / `vina.exe`)
2. `VINA_BINARY`
3. `--vina /path/to/vina`
4. `tools/vina.exe`

Windows: download from [AutoDock-Vina releases](https://github.com/ccsb-scripps/AutoDock-Vina/releases), rename to `vina.exe`, place under `tools/`.

---

## Inference API (no physics stack)

Uses only `xgboost_model.pkl` and `features.csv`. Do **not** install OpenMM / Vina in the API image.

### Local (conda)

```bash
conda activate protein
# after offline training has produced models/ + data/processed/
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness |
| `POST` | `/predict` | Body `{"mutation": "L858R"}` → predicted Vina score |
| `GET` | `/explain/{mutation}` | SHAP feature contributions |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"mutation\": \"L858R\"}"
curl http://localhost:8000/explain/L858R
```

Optional env overrides: `MODEL_PATH`, `FEATURES_PATH`.

### Docker (inference-only)

Installs `requirements-api.txt` only (FastAPI, XGBoost, SHAP, pandas):

```bash
# from repo root, after training
docker compose up --build
```

Compose mounts `models/`, `data/processed/`, and `configs/` read-only. Docs: http://localhost:8000/docs

---

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
