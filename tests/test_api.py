"""API tests for Step 6 FastAPI inference endpoints."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBRegressor

from src.api import create_app
from src.explain import build_tree_explainer, shap_contributions

FEATURE_NAMES = [
    "residue_position",
    "dist_to_ligand",
    "dist_to_pocket",
    "sasa",
    "delta_charge",
    "delta_volume",
]


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
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
    ]
    features = pd.DataFrame(rows)
    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False)

    X = features[FEATURE_NAMES]
    y = features["vina_score"].astype(float)
    model = XGBRegressor(
        max_depth=2,
        n_estimators=10,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y)

    model_path = tmp_path / "xgboost_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "feature_names": FEATURE_NAMES}, handle)

    return model_path, features_path


@pytest.fixture
def api_client(tmp_path: Path):
    model_path, features_path = _write_artifacts(tmp_path)
    app = create_app(model_path=model_path, features_path=features_path)
    with TestClient(app) as client:
        yield client


def test_health(api_client: TestClient):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_ok(api_client: TestClient):
    response = api_client.post("/predict", json={"mutation": "L858R"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["mutation"] == "L858R"
    assert isinstance(payload["predicted_vina_score"], float)
    assert set(payload["features"]) == set(FEATURE_NAMES)


def test_predict_unknown_mutation(api_client: TestClient):
    response = api_client.post("/predict", json={"mutation": "X999Y"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_explain_ok(api_client: TestClient):
    response = api_client.get("/explain/T790M")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mutation"] == "T790M"
    assert isinstance(payload["predicted_vina_score"], float)
    assert isinstance(payload["base_value"], float)
    assert set(payload["shap_values"]) == set(FEATURE_NAMES)
    # Additive SHAP reconstruction should match the model prediction closely
    assert payload["prediction_from_shap"] == pytest.approx(
        payload["predicted_vina_score"],
        abs=1e-4,
    )


def test_explain_unknown_mutation(api_client: TestClient):
    response = api_client.get("/explain/X999Y")
    assert response.status_code == 404


def test_shap_contributions_deterministic(tmp_path: Path):
    model_path, features_path = _write_artifacts(tmp_path)
    with model_path.open("rb") as handle:
        artifact = pickle.load(handle)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    store = pd.read_csv(features_path).set_index("mutation")
    row = store.loc[["L858R"], feature_names]

    explainer = build_tree_explainer(model)
    first = shap_contributions(explainer, row, feature_names)
    second = shap_contributions(explainer, row, feature_names)
    assert first["shap_values"] == second["shap_values"]
    pred = float(model.predict(row)[0])
    assert first["prediction_from_shap"] == pytest.approx(pred, abs=1e-4)
    assert abs(sum(first["shap_values"].values())) > 0 or first["base_value"] == pytest.approx(pred)
