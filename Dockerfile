FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.3.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-ansi

COPY src/telco_churn ./src/telco_churn
COPY models/churn_model.pkl ./models/churn_model.pkl
COPY data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv ./data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p "$MPLCONFIGDIR" \
    && chown -R appuser:appuser /app "$MPLCONFIGDIR"

USER appuser

CMD ["sh", "-c", "uvicorn src.telco_churn.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
