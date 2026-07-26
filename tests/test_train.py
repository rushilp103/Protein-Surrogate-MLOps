"""Unit tests for Step 7 training (mutation index, GroupKFold, GridSearch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.train import (
    TrainError,
    assert_group_folds_disjoint,
    fit_final_model,
    load_aligned_dataset,
    prediction_table,
    run_training,
    tune_surrogate,
)


FEATURE_ROWS = [
    {
        "mutation": "L858R",
        "residue_position": 858,
        "wt_aa": "L",
        "mut_aa": "R",
        "dist_to_ligand": 5.0,
        "dist_to_pocket": 3.0,
        "sasa": 50.0,
        "delta_charge": 1.0,
        "delta_volume": 6.7,
        "vina_score": -7.5,
    },
    {
        "mutation": "T790M",
        "residue_position": 790,
        "wt_aa": "T",
        "mut_aa": "M",
        "dist_to_ligand": 6.0,
        "dist_to_pocket": 4.0,
        "sasa": 40.0,
        "delta_charge": 0.0,
        "delta_volume": 46.8,
        "vina_score": -8.0,
    },
    {
        "mutation": "G719S",
        "residue_position": 719,
        "wt_aa": "G",
        "mut_aa": "S",
        "dist_to_ligand": 7.0,
        "dist_to_pocket": 5.0,
        "sasa": 60.0,
        "delta_charge": 0.0,
        "delta_volume": 28.9,
        "vina_score": -6.5,
    },
    {
        "mutation": "G719A",
        "residue_position": 719,
        "wt_aa": "G",
        "mut_aa": "A",
        "dist_to_ligand": 7.1,
        "dist_to_pocket": 5.1,
        "sasa": 55.0,
        "delta_charge": 0.0,
        "delta_volume": 25.0,
        "vina_score": -6.8,
    },
    {
        "mutation": "L861Q",
        "residue_position": 861,
        "wt_aa": "L",
        "mut_aa": "Q",
        "dist_to_ligand": 8.0,
        "dist_to_pocket": 6.0,
        "sasa": 45.0,
        "delta_charge": 0.0,
        "delta_volume": 10.0,
        "vina_score": -7.1,
    },
    {
        "mutation": "S768I",
        "residue_position": 768,
        "wt_aa": "S",
        "mut_aa": "I",
        "dist_to_ligand": 9.0,
        "dist_to_pocket": 7.0,
        "sasa": 35.0,
        "delta_charge": 0.0,
        "delta_volume": 30.0,
        "vina_score": -7.3,
    },
]


def _write_feature_store(tmp_path: Path) -> tuple[Path, Path]:
    features = pd.DataFrame(FEATURE_ROWS)
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"
    features.to_csv(features_path, index=False)
    features[["mutation", "vina_score"]].to_csv(labels_path, index=False)
    return features_path, labels_path


def test_load_aligned_dataset_sets_mutation_index(tmp_path: Path):
    features_path, labels_path = _write_feature_store(tmp_path)
    X, y, groups = load_aligned_dataset(features_path, labels_path)

    assert list(X.index) == [row["mutation"] for row in FEATURE_ROWS]
    assert list(y.index) == list(X.index)
    assert list(groups.index) == list(X.index)
    assert "mutation" not in X.columns
    assert "wt_aa" not in X.columns
    assert "mut_aa" not in X.columns
    assert "vina_score" not in X.columns
    assert "residue_position" in X.columns
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)
    assert y.loc["L858R"] == pytest.approx(-7.5)


def test_groupkfold_train_test_groups_disjoint(tmp_path: Path):
    features_path, labels_path = _write_feature_store(tmp_path)
    _X, _y, groups = load_aligned_dataset(features_path, labels_path)

    assert_group_folds_disjoint(groups, n_splits=3)

    dummy_X = np.zeros((len(groups), 1))
    dummy_y = np.zeros(len(groups))
    for train_idx, test_idx in GroupKFold(n_splits=3).split(dummy_X, dummy_y, groups=groups):
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))


def test_model_predictions_are_deterministic(tmp_path: Path):
    features_path, labels_path = _write_feature_store(tmp_path)
    X, y, _groups = load_aligned_dataset(features_path, labels_path)
    model = fit_final_model(
        X,
        y,
        {"max_depth": 3, "n_estimators": 20, "learning_rate": 0.1},
        random_state=42,
    )
    first = model.predict(X)
    second = model.predict(X)
    np.testing.assert_allclose(first, second)


def test_prediction_table_preserves_mutation_labels(tmp_path: Path):
    features_path, labels_path = _write_feature_store(tmp_path)
    _X, y, _groups = load_aligned_dataset(features_path, labels_path)
    table = prediction_table(y, y.to_numpy())
    assert list(table["mutation"]) == list(y.index)
    assert table["actual"].tolist() == y.tolist()


def test_tune_surrogate_returns_best_params(tmp_path: Path):
    features_path, labels_path = _write_feature_store(tmp_path)
    X, y, groups = load_aligned_dataset(features_path, labels_path)
    search = tune_surrogate(
        X,
        y,
        groups,
        n_splits=3,
        random_state=42,
        param_grid={
            "max_depth": [2, 3],
            "n_estimators": [10, 20],
            "learning_rate": [0.1],
        },
    )
    assert "max_depth" in search.best_params_
    assert "n_estimators" in search.best_params_
    assert search.best_estimator_ is not None


def test_run_training_writes_artifacts(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_feature_store(processed)

    config = {
        "protein": {
            "name": "EGFR",
            "uniprot_id": "P00533",
            "alphafold_id": "AF-P00533-F1",
        },
        "drug": {"name": "Gefitinib", "pubchem_cid": 1, "smiles": "C"},
        "mutations": [row["mutation"] for row in FEATURE_ROWS],
        "paths": {
            "raw": "raw",
            "mutants": "mutants",
            "docking": "docking",
            "processed": "processed",
            "models": "models",
            "outputs": "outputs",
        },
        "training": {
            "target_column": "vina_score",
            "group_column": "residue_position",
            "n_splits": 3,
            "test_size": 0.3,
            "random_state": 42,
        },
    }

    import src.train as train_mod

    original_grid = train_mod.DEFAULT_PARAM_GRID
    train_mod.DEFAULT_PARAM_GRID = {
        "max_depth": [2],
        "n_estimators": [10],
        "learning_rate": [0.1],
    }
    try:
        artifacts = run_training(config, root=tmp_path)
    finally:
        train_mod.DEFAULT_PARAM_GRID = original_grid

    assert artifacts["model"].is_file()
    assert artifacts["figure"].is_file()
    assert artifacts["predictions"].is_file()
    assert artifacts["metrics"].is_file()

    pred = pd.read_csv(artifacts["predictions"])
    assert {"mutation", "actual", "predicted"} <= set(pred.columns)
    assert len(pred) >= 1


def test_missing_features_raise(tmp_path: Path):
    with pytest.raises(TrainError, match="features.csv"):
        load_aligned_dataset(tmp_path / "missing.csv", tmp_path / "also_missing.csv")
