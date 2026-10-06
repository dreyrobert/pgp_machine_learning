"""Carregamento do modelo versionado e do histórico observado."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/modelo_acidentes_rf_baseline_v1.joblib"
METADATA_PATH = ROOT / "models/modelo_acidentes_rf_baseline_v1_metadata.json"
DATA_PATH = ROOT / "data/processed/snapshot_acidentes.parquet"


def load_model():
    payload = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if payload["metadata"] != metadata:
        raise ValueError("Os metadados externos não correspondem ao modelo versionado.")
    expected = metadata["numeric_features"] + metadata["categorical_features"]
    if payload["feature_columns"] != expected:
        raise ValueError("As colunas do modelo não correspondem aos metadados.")
    return payload


def load_history(metadata):
    history = pd.read_parquet(DATA_PATH, columns=["municipio", "data_referencia", "qtd_acidentes"])
    history["data_referencia"] = pd.to_datetime(history.data_referencia)
    # O snapshot contém meses futuros preenchidos com zero: não são observações.
    history = history.loc[history.data_referencia.between(metadata["training_start_month"], metadata["cutoff_month"])].copy()
    if history.empty or history.duplicated(["municipio", "data_referencia"]).any():
        raise ValueError("Base histórica vazia ou com meses duplicados.")
    if history.qtd_acidentes.isna().any() or (history.qtd_acidentes < 0).any():
        raise ValueError("Contagens históricas inválidas.")
    if history.data_referencia.max() != pd.Timestamp(metadata["cutoff_month"]):
        raise ValueError("A base não contém o mês final informado nos metadados.")
    return history.sort_values(["municipio", "data_referencia"]).reset_index(drop=True)


def predict(payload, features):
    expected = payload["feature_columns"]
    if list(features.columns) != expected or len(features) != 1:
        raise ValueError("Entrada incompatível com as colunas do modelo.")
    numeric = features[payload["numeric_features"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Os atributos numéricos devem ser finitos.")
    result = float(payload["model"].predict(features)[0])
    if not np.isfinite(result):
        raise ValueError("O modelo retornou uma previsão inválida.")
    return max(0.0, result)
