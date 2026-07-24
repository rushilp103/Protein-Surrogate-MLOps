# Protein-Surrogate-ML — inference image (filled in Step 6)
# Offline FoldX / AutoDock Vina tooling is intentionally excluded.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
