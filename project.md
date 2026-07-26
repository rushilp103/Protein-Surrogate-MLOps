# Project: Structure-Aware Machine Learning Surrogate Model for Accelerated Protein Variant Screening

## 1. Project Overview

**Core Research Question:** Can machine learning predict molecular docking scores from AlphaFold-derived structural and biochemical features, eliminating the need to run computationally expensive docking simulations for every new protein variant?

**Motivation:** Molecular docking estimates protein–ligand interactions but is a computationally heavy process. The traditional workflow requires generating a structure, running a docking simulation, and obtaining a score for *every* candidate mutation. This project replaces that bottleneck with an XGBoost surrogate model that predicts AutoDock Vina scores instantly based on precomputed structural and biochemical features.

**Environment requirement:** All terminal commands for this project **must** run under an activated `protein` conda environment. Do not use the system Python, another conda env, a separate `venv`, or a bare path to `python.exe` without activation.

```bash
conda activate protein
```

Activation is required (not optional): it puts the env on `PATH`, including `Library/bin` (e.g. `libiomp5md.dll` for MKL/OpenMM/NumPy). Skipping activation can cause native DLL load failures on Windows. After activating, run all `python`, `pytest`, and pipeline commands in that same shell.

**Repository:** GitHub repo is already set up at [https://github.com/rushilp103/Protein-Surrogate-ML](https://github.com/rushilp103/Protein-Surrogate-ML).

---



## 2. System Architecture

The project is strictly divided into two distinct environments to ensure production readiness and a lightweight deployment footprint.

### 2.1. Offline Pipeline (Data Generation & Training)

This heavy computational pipeline runs locally or on a high-performance server.

1. **Data Ingestion:** AlphaFold WT Structure + Drug SDF + Mutation List.
2. **Structural Modeling:** OpenMM and PDBFixer to mutate residues, add missing hydrogens, and perform AMBER local energy minimization to resolve atomic clashes in mutant structures.
3. **Label Generation:** AutoDock Vina runs simulations to find the top binding pose (Top 1) and affinity score.
4. **Feature Extraction:** Biopython and FreeSASA convert 3D structures into 1D tabular features.
5. **Model Training:** XGBoost learns the mapping between structural features and Vina scores using `GroupKFold` cross-validation.



### 2.2. Online Pipeline (FastAPI Inference)

A lightweight Dockerized environment strictly for model inference.

- **Feature Store:** Precomputed features are loaded into memory.
- **FastAPI:** Exposes `/predict` and `/explain` endpoints.
- **No Physics Engine:** OpenMM and AutoDock Vina are completely excluded from the Docker image to maintain a small container size (<500MB).

---



## 3. Technology Stack


| Category                | Component / Tool                 | Primary Use                                          |
| ----------------------- | -------------------------------- | ---------------------------------------------------- |
| **Protein Processing**  | Biopython, AlphaFold DB          | Downloading structures, parsing PDB files.           |
| **Mutation Modeling**   | OpenMM, PDBFixer                 | Residue substitution and AMBER energy minimization.  |
| **Molecular Docking**   | AutoDock Vina, RDKit / OpenBabel | Preparing PDBQT files and simulating binding.        |
| **Feature Engineering** | FreeSASA, MDTraj, NumPy          | Calculating SASA, distances, and biochemical shifts. |
| **Machine Learning**    | XGBoost, Scikit-Learn            | Training the surrogate model, evaluating metrics.    |
| **Explainability**      | SHAP                             | Generating feature importance and force plots.       |
| **MLOps / Backend**     | FastAPI, Uvicorn, Docker         | Serving predictions via REST API.                    |
| **Testing**             | pytest                           | Unit and integration testing across the pipeline.    |


---



## 4. Repository Layout

```
protein-surrogate/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
│
├── src/
│   ├── download.py
│   ├── openmm_mutator.py
│   ├── docking.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── api.py
│
├── data/
│   ├── raw/
│   ├── mutants/
│   ├── docking/
│   └── processed/
│
├── models/
│   └── xgboost_model.pkl
│
├── outputs/
│   ├── metrics/
│   ├── figures/
│   └── reports/
│
└── tests/
```

---



## 5. Implementation Plan: Sized Steps



### Step 1: Scaffolding and Configuration

**Objective:** Set up the repository and the YAML configuration parser to drive the pipeline.

- **Libraries:** `pyyaml`, `pandas`.
- **Action Items:**
  1. Use the existing `Protein-Surrogate-ML` GitHub repository and create the required directory structure (see Repository Layout).
  2. Use the existing `protein` conda environment and install baseline dependencies there (all terminal work stays in this env).
  3. Create `configs/egfr.yaml` and hardcode a micro-dataset of 3 mutations for initial testing (e.g., `L858R`, `T790M`, `G719S`).



### Step 2: Wild-Type (WT) Preparation

**Objective:** Programmatically download and clean the baseline biological data.

- **Libraries:** `biopython`, `rdkit`, `urllib`.
- **Action Items:**
  1. Write `src/download.py` to retrieve the AlphaFold PDB for EGFR and the SDF for Gefitinib.
  2. Clean the WT structure (removing heteroatoms or water) to prepare it for mutation.
  3. Convert the drug's `.sdf` file to a `.pdbqt` file using RDKit or OpenBabel.



### Step 3: The Physics Engine (OpenMM & Vina)

**Objective:** Automate the computationally heavy 3D modeling and docking.

- **Libraries:** `subprocess`, `os`, `re`, `openmm`,`pdbfixer`.
- **Action Items:**
  1. Write `src/openmm_mutator.py` to loop through the mutation list. Use PDBFixer to apply the mutation and add missing hydrogens at pH 7.0. Then, run an OpenMM AMBER14 local energy minimization to resolve steric clashes and save the mutant `.pdb`.
  2. Write `src/docking.py` to automate the Vina CLI.
  3. Implement a regex parser to extract the Top-1 most negative binding affinity score from the Vina output log. Save this as `labels.csv`.



### Step 4: Feature Extraction (The Math Engine)

**Objective:** Translate the 3D structures into tabular ML data.

- **Libraries:** `biopython`, `freesasa`, `pandas`, `numpy`.
- **Action Items:**
  1. Write `src/features.py` to parse each generated mutant PDB.
  2. Calculate structural features (distance to ligand, distance to pocket, SASA).
  3. Calculate biochemical changes (charge change $\Delta$, volume change).
  4. Merge these calculated features with the Vina scores from Step 3 to output the final `features.csv` (The Feature Store).



### Step 5: Machine Learning (The Brain)

**Objective:** Train a robust surrogate model preventing data leakage.

- **Libraries:** `scikit-learn`, `xgboost`, `shap`, `matplotlib`.
- **Action Items:**
  1. In `src/train.py`, load `features.csv` and set the target variable to the Vina Score.
  2. Implement `GroupKFold` cross-validation where `group = residue position` to ensure the model generalizes to unseen regions of the protein.
  3. Train the `XGBRegressor` and export as `xgboost_model.pkl`.
  4. Generate SHAP summary plots in Jupyter Notebooks to confirm biological logic (e.g., verifying that distance to the pocket drives predictions).



### Step 6: Deployment & API Design

**Objective:** Wrap the trained model in a lightweight REST API.

- **Libraries:** `fastapi`, `uvicorn`, `pydantic`.
- **Action Items:**
  1. Write `src/api.py`. Use FastAPI startup events to load `xgboost_model.pkl` and `features.csv` into memory.
  2. Implement `POST /predict`. It should look up the mutation in the feature store, construct the feature vector, and return the predicted score.
  3. Implement `GET /explain/{mutation}` to return SHAP contribution values.
  4. Write a lightweight `Dockerfile` strictly for the Python/FastAPI environment.



### Step 7: Scale and Document

**Objective:** Scale up to the full dataset and finalize repository.

- **Action Items:**
  1. Update `configs/egfr.yaml` to include 50–200 mutations.
  2. Re-enable data splitting: Remove the dry-run logic (training and predicting on the same data). Re-implement the K-Fold Cross Validation to properly evaluate the model on unseen data.
  3. Uncap model hyperparameters: Remove the remporary micro-dataset constraints (max_depth=2, n_estimators=10). Implement GridSearchCV to find optimal parameters for the macro-dataset (e.g., testing max_depth 4-6, n_estimators 100-300).
  4. Preserve mutation index: Modify the pandas preprocessing so it no longer does the mutation column. Instead, use X = X.set_index('mutation') (and drop wt_aa, mut_aa). This keeps the matrix strictly numeric for XGBoost while allowing us to track which prediction belongs to which mutant in the final output table.
  5. Run the offline pipeline overnight to compute the full `features.csv` and train the final model.
  6. Update `README.md` with explicit instructions on running the data generation pipeline vs. spinning up the Dockerized API.

---



## 6. Testing Strategy

To ensure code resilience and biological validity, utilize `pytest` to implement the following test suites:

### Unit Tests

- **Config Parser:** Assert that `egfr.yaml` loads correctly and raises errors for missing required fields (like `protein` or `drug`).
- **Biochemical Calculators:** Write tests asserting that a known mutation (e.g., L to R) returns the mathematically correct change in charge and volume.
- **Log Parsing:** Pass a dummy Vina text log into your parser and `assert` that it correctly extracts the Top-1 negative score as a float.



### Integration Tests

- **Feature Pipeline:** Run a single mutation through `features.py` and `assert` the resulting DataFrame has zero `NaN` values and matches the expected column schema.
- **API Tests:** Use FastAPI's `TestClient` to send a mocked request to `POST /predict` and `assert` it returns a `200 OK` status and a valid JSON structure.



### ML Tests (Sanity Checks)

- **Deterministic Output:** `assert` that passing the exact same feature vector to the model twice yields the exact same predicted docking score.
- **Generalization Check:** `assert` that the training routine successfully isolates residue groups using `GroupKFold` by verifying the intersection of train and test group IDs is empty.

