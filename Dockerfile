# Inference image — FastAPI + XGBoost only.
# Offline physics tooling (OpenMM, AutoDock Vina, RDKit, FreeSASA) is excluded.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/xgboost_model.pkl \
    FEATURES_PATH=/app/data/processed/features.csv

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
