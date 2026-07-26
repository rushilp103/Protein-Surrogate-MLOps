"""XGBoost surrogate model training (Step 5 / Step 7 — macro-dataset)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from xgboost import XGBRegressor

from src.config import load_config, resolve_paths

DROP_FEATURE_COLUMNS = ("wt_aa", "mut_aa")
MODEL_FILENAME = "xgboost_model.pkl"
IMPORTANCE_FIGURE = "feature_importances.png"
METRICS_FILENAME = "training_metrics.json"
PREDICTIONS_FILENAME = "actual_vs_predicted.csv"

DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "max_depth": [4, 5, 6],
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
}


class TrainError(RuntimeError):
    """Raised when training inputs are missing or misaligned."""


def load_aligned_dataset(
    features_path: Path,
    labels_path: Path,
    target_column: str = "vina_score",
    group_column: str = "residue_position",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load features and labels, merge on ``mutation``, and build X / y / groups.

    ``mutation`` becomes the DataFrame index (traceable predictions without
    feeding a string column to XGBoost). ``wt_aa`` / ``mut_aa`` are dropped;
    ``group_column`` stays in ``X`` as a numeric feature and is also returned
    separately for group-aware splitting.
    """
    if not features_path.is_file():
        raise TrainError(f"features.csv not found: {features_path}")
    if not labels_path.is_file():
        raise TrainError(f"labels.csv not found: {labels_path}")

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    if "mutation" not in features.columns:
        raise TrainError("features.csv is missing a 'mutation' column")
    if "mutation" not in labels.columns:
        raise TrainError("labels.csv is missing a 'mutation' column")
    if target_column not in labels.columns:
        raise TrainError(f"labels.csv is missing target column '{target_column}'")
    if group_column not in features.columns:
        raise TrainError(f"features.csv is missing group column '{group_column}'")

    # Target must come from labels only (avoid leakage / column clash on merge).
    features = features.drop(columns=[target_column], errors="ignore")

    merged = features.merge(labels, on="mutation", how="inner", validate="one_to_one")
    if merged.empty:
        raise TrainError("Merge of features.csv and labels.csv produced zero rows")

    missing = [c for c in ("mutation", *DROP_FEATURE_COLUMNS) if c not in merged.columns]
    if missing:
        raise TrainError(f"Merged frame missing expected columns: {', '.join(missing)}")

    indexed = merged.set_index("mutation")
    y = indexed[target_column].astype(float)
    groups = indexed[group_column].copy()
    X = indexed.drop(columns=[*DROP_FEATURE_COLUMNS, target_column])

    if X.empty or X.shape[1] == 0:
        raise TrainError("Feature matrix X is empty after dropping identifier columns")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        raise TrainError(f"Non-numeric feature columns remain: {', '.join(non_numeric)}")

    return X, y, groups


def _resolve_n_splits(n_splits: int, n_groups: int) -> int:
    """Clamp fold count so GroupKFold has enough distinct groups."""
    if n_groups < 2:
        raise TrainError(
            f"Need at least 2 residue groups for cross-validation; found {n_groups}"
        )
    return max(2, min(int(n_splits), n_groups))


def tune_surrogate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    param_grid: dict[str, list[Any]] | None = None,
) -> GridSearchCV:
    """
    Grid-search XGBRegressor hyperparameters with GroupKFold CV.

    Groups are residue positions so the same site never appears in both
    train and validation folds.
    """
    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID
    n_splits = _resolve_n_splits(n_splits, int(groups_train.nunique()))
    cv = GroupKFold(n_splits=n_splits)
    search = GridSearchCV(
        estimator=XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        ),
        param_grid=grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search


def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, Any],
    *,
    random_state: int = 42,
) -> XGBRegressor:
    """Refit an XGBRegressor on the full dataset with tuned hyperparameters."""
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,
        **best_params,
    )
    model.fit(X, y)
    return model


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R² (R² may be omitted when undefined)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    metrics: dict[str, float] = {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
    }
    if len(y_true_arr) >= 2:
        r2 = float(r2_score(y_true_arr, y_pred_arr))
        if np.isfinite(r2):
            metrics["r2"] = r2
    return metrics


def prediction_table(
    y_true: pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Build an Actual vs Predicted table keyed by mutation index."""
    return pd.DataFrame(
        {
            "mutation": y_true.index.astype(str),
            "actual": y_true.to_numpy(),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    )


def plot_feature_importances(
    model: XGBRegressor,
    feature_names: list[str],
    out_path: Path,
) -> Path:
    """Bar plot of XGBoost gain-based feature importances."""
    importances = model.feature_importances_
    order = importances.argsort()[::-1]
    names = [feature_names[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(range(len(names)), values[::-1], color="#2a6f97")
    ax.set_yticks(range(len(names)), labels=names[::-1])
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title("XGBoost feature importances")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def assert_group_folds_disjoint(groups: pd.Series, n_splits: int) -> None:
    """Sanity check: GroupKFold train/test group IDs never overlap."""
    n_splits = _resolve_n_splits(n_splits, int(groups.nunique()))
    dummy_X = np.zeros((len(groups), 1))
    dummy_y = np.zeros(len(groups))
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(
        dummy_X, dummy_y, groups=groups
    ):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        if train_groups & test_groups:
            raise TrainError(
                "GroupKFold leakage detected: "
                f"{sorted(train_groups & test_groups)}"
            )


def run_training(config: dict[str, Any], root: Path | None = None) -> dict[str, Path]:
    """Tune on a group hold-out, refit on all data, and write artifacts."""
    paths = resolve_paths(config, root=root)
    processed_dir = paths["processed"]
    models_dir = paths["models"]
    outputs_dir = paths["outputs"]
    figures_dir = outputs_dir / "figures"
    metrics_dir = outputs_dir / "metrics"

    training = config.get("training") or {}
    target_column = str(training.get("target_column", "vina_score"))
    group_column = str(training.get("group_column", "residue_position"))
    n_splits = int(training.get("n_splits", 5))
    random_state = int(training.get("random_state", 42))
    test_size = float(training.get("test_size", 0.2))

    features_path = processed_dir / "features.csv"
    labels_path = processed_dir / "labels.csv"

    X, y, groups = load_aligned_dataset(
        features_path,
        labels_path,
        target_column=target_column,
        group_column=group_column,
    )

    assert_group_folds_disjoint(groups, n_splits)

    print(
        f"rows: {len(X)}  features: {list(X.columns)}  "
        f"groups: {int(groups.nunique())}",
        flush=True,
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    print(
        f"hold-out split: train={len(X_train)} test={len(X_test)} "
        f"(GroupShuffleSplit test_size={test_size})",
        flush=True,
    )
    print("Hunting for optimal hyperparameters (GroupKFold GridSearchCV)...", flush=True)

    search = tune_surrogate(
        X_train,
        y_train,
        groups_train,
        n_splits=n_splits,
        random_state=random_state,
    )
    best_params = dict(search.best_params_)
    cv_rmse = float(np.sqrt(-search.best_score_))
    print(f"best_params: {best_params}", flush=True)
    print(f"best_cv_rmse: {cv_rmse:.4f}", flush=True)

    y_test_pred = search.best_estimator_.predict(X_test)
    test_metrics = regression_metrics(y_test, y_test_pred)
    r2_msg = (
        f"r2={test_metrics['r2']:.4f}"
        if "r2" in test_metrics
        else "r2=n/a"
    )
    print(
        f"hold-out  rmse={test_metrics['rmse']:.4f}  "
        f"mae={test_metrics['mae']:.4f}  {r2_msg}",
        flush=True,
    )

    print("refitting final model on full dataset with best_params...", flush=True)
    model = fit_final_model(X, y, best_params, random_state=random_state)

    table = prediction_table(y_test, y_test_pred)
    print("\nHold-out Actual vs Predicted:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(flush=True)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "feature_names": list(X.columns),
                "best_params": best_params,
            },
            handle,
        )

    fig_path = plot_feature_importances(
        model, list(X.columns), figures_dir / IMPORTANCE_FIGURE
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    table_path = metrics_dir / PREDICTIONS_FILENAME
    table.to_csv(table_path, index=False, float_format="%.6f")

    metrics_payload = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_groups": int(groups.nunique()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": test_size,
        "n_splits": _resolve_n_splits(n_splits, int(groups_train.nunique())),
        "best_params": best_params,
        "best_cv_rmse": cv_rmse,
        "holdout": test_metrics,
        "feature_names": list(X.columns),
    }
    metrics_path = metrics_dir / METRICS_FILENAME
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    return {
        "model": model_path,
        "figure": fig_path,
        "predictions": table_path,
        "metrics": metrics_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train XGBoost surrogate with GroupShuffleSplit hold-out and "
            "GroupKFold GridSearchCV, then refit on all rows."
        ),
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/egfr.yaml",
        help="Path to pipeline YAML config (default: configs/egfr.yaml)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root for resolving relative paths (default: cwd)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root if args.root is not None else Path.cwd()
    config = load_config(args.config)
    try:
        artifacts = run_training(config, root=root)
    except TrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"model:       {artifacts['model']}")
    print(f"figure:      {artifacts['figure']}")
    print(f"predictions: {artifacts['predictions']}")
    print(f"metrics:     {artifacts['metrics']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
