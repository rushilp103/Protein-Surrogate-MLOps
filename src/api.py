"""FastAPI inference service (Step 6).

Loads ``xgboost_model.pkl`` and the feature store into memory at startup.
OpenMM / AutoDock Vina are not used here — only tabular lookup + XGBoost/SHAP.
"""

from __future__ import annotations

import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

from src.explain import build_tree_explainer, shap_contributions

DEFAULT_MODEL_PATH = Path("models/xgboost_model.pkl")
DEFAULT_FEATURES_PATH = Path("data/processed/features.csv")


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``."""

    mutation: str = Field(..., min_length=1, examples=["L858R"])


class PredictResponse(BaseModel):
    mutation: str
    predicted_vina_score: float
    features: dict[str, float]


class ExplainResponse(BaseModel):
    mutation: str
    predicted_vina_score: float
    base_value: float
    shap_values: dict[str, float]
    prediction_from_shap: float


def _resolve_artifact_paths() -> tuple[Path, Path]:
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    features_path = Path(os.environ.get("FEATURES_PATH", DEFAULT_FEATURES_PATH))
    return model_path, features_path


def load_model_artifact(path: Path) -> tuple[XGBRegressor, list[str]]:
    """Load the pickled ``{"model", "feature_names"}`` training artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or "model" not in artifact or "feature_names" not in artifact:
        raise ValueError(
            f"Unexpected model format in {path}; expected dict with 'model' and 'feature_names'"
        )
    return artifact["model"], list(artifact["feature_names"])


def load_feature_store(path: Path) -> pd.DataFrame:
    """Load ``features.csv`` and index rows by mutation string."""
    if not path.is_file():
        raise FileNotFoundError(f"Feature store not found: {path}")
    frame = pd.read_csv(path)
    if "mutation" not in frame.columns:
        raise ValueError("Feature store is missing a 'mutation' column")
    if frame["mutation"].duplicated().any():
        raise ValueError("Feature store contains duplicate mutation keys")
    return frame.set_index("mutation", drop=False)


def feature_vector_for_mutation(
    store: pd.DataFrame,
    mutation: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """Look up a mutation and return a 1-row DataFrame in model feature order."""
    if mutation not in store.index:
        raise KeyError(mutation)
    row = store.loc[mutation]
    missing = [name for name in feature_names if name not in store.columns]
    if missing:
        raise ValueError(f"Feature store missing columns required by model: {', '.join(missing)}")
    values = {name: float(row[name]) for name in feature_names}
    return pd.DataFrame([values], columns=feature_names)


def create_app(
    model_path: Path | None = None,
    features_path: Path | None = None,
) -> FastAPI:
    """
    Build the FastAPI app.

    Paths default to ``MODEL_PATH`` / ``FEATURES_PATH`` env vars, then repo-relative
    defaults under ``models/`` and ``data/processed/``.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_model, resolved_features = _resolve_artifact_paths()
        if model_path is not None:
            resolved_model = model_path
        if features_path is not None:
            resolved_features = features_path

        model, feature_names = load_model_artifact(resolved_model)
        store = load_feature_store(resolved_features)
        explainer = build_tree_explainer(model)

        app.state.model = model
        app.state.feature_names = feature_names
        app.state.feature_store = store
        app.state.explainer = explainer
        yield

    app = FastAPI(
        title="Protein Surrogate MLOps API",
        description=(
            "Predict AutoDock Vina docking scores from precomputed structural "
            "features using a trained XGBoost surrogate."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest) -> PredictResponse:
        mutation = body.mutation.strip()
        try:
            vector = feature_vector_for_mutation(
                app.state.feature_store,
                mutation,
                app.state.feature_names,
            )
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Mutation '{mutation}' not found in feature store",
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        score = float(app.state.model.predict(vector)[0])
        return PredictResponse(
            mutation=mutation,
            predicted_vina_score=score,
            features={name: float(vector.iloc[0][name]) for name in app.state.feature_names},
        )

    @app.get("/explain/{mutation}", response_model=ExplainResponse)
    def explain(mutation: str) -> ExplainResponse:
        mutation = mutation.strip()
        try:
            vector = feature_vector_for_mutation(
                app.state.feature_store,
                mutation,
                app.state.feature_names,
            )
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Mutation '{mutation}' not found in feature store",
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        score = float(app.state.model.predict(vector)[0])
        shap_result = shap_contributions(
            app.state.explainer,
            vector,
            app.state.feature_names,
        )
        return ExplainResponse(
            mutation=mutation,
            predicted_vina_score=score,
            base_value=shap_result["base_value"],
            shap_values=shap_result["shap_values"],
            prediction_from_shap=shap_result["prediction_from_shap"],
        )

    return app


app = create_app()
