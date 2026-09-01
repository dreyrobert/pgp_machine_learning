from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "qtd_acidentes"
MODEL_VERSION = "v1"
MODEL_NAME = "RandomForestRegressor"
FEATURE_SET_NAME = "calendario_lags_municipio"

LEAKAGE_CURRENT_MONTH_COLUMNS = [
    "qtd_mortos",
    "qtd_feridos_graves",
    "qtd_feridos_leves",
    "total_veiculos_envolvidos",
    "taxa_severidade_acumulada",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def month_start(value: pd.Timestamp) -> pd.Timestamp:
    return value.to_period("M").to_timestamp()


def detect_last_complete_month(root: Path) -> pd.Timestamp:
    interim_path = root / "data" / "interim" / "acidentes_SC.parquet"
    acidentes = pd.read_parquet(interim_path, columns=["data_inversa"])
    acidentes["data_inversa"] = pd.to_datetime(acidentes["data_inversa"])
    return month_start(acidentes["data_inversa"].max())


def load_snapshot(root: Path, cutoff_month: pd.Timestamp | None = None) -> pd.DataFrame:
    snapshot_path = root / "data" / "processed" / "snapshot_acidentes.parquet"
    data = pd.read_parquet(snapshot_path)
    data["data_referencia"] = pd.to_datetime(data["data_referencia"])
    data = data.sort_values(["municipio", "data_referencia"]).reset_index(drop=True)

    if cutoff_month is not None:
        data = data[data["data_referencia"] <= cutoff_month].copy()

    return data.reset_index(drop=True)


def add_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy().sort_values(["municipio", "data_referencia"]).reset_index(drop=True)

    data["tempo_idx"] = (data["ano"] - data["ano"].min()) * 12 + data["mes"]
    data["mes_sin"] = np.sin(2 * np.pi * data["mes"] / 12)
    data["mes_cos"] = np.cos(2 * np.pi * data["mes"] / 12)

    historical_current_month_columns = [
        "qtd_mortos",
        "qtd_feridos_graves",
        "qtd_feridos_leves",
        "total_veiculos_envolvidos",
    ]

    for col in historical_current_month_columns:
        shifted = data.groupby("municipio")[col].shift(1)
        data[f"{col}_lag_1m"] = shifted.fillna(0)
        data[f"{col}_media_3m"] = (
            shifted.groupby(data["municipio"])
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            .fillna(0)
        )
        data[f"{col}_media_6m"] = (
            shifted.groupby(data["municipio"])
            .transform(lambda s: s.rolling(window=6, min_periods=1).mean())
            .fillna(0)
        )

    data["acidentes_acum_ano_ate_mes_anterior"] = (
        data.groupby(["municipio", "ano"])[TARGET].cumsum() - data[TARGET]
    )

    data["media_historica_municipio"] = (
        data.groupby("municipio")[TARGET]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        .fillna(0)
    )

    return data


def feature_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    calendar_features = [
        "ano",
        "mes",
        "trimestre",
        "eh_alta_temporada",
        "tempo_idx",
        "mes_sin",
        "mes_cos",
    ]

    accident_lag_features = [
        "lag_acidentes_1m",
        "lag_acidentes_2m",
        "lag_acidentes_12m",
        "media_movel_3m",
        "media_movel_6m",
        "acidentes_acum_ano_ate_mes_anterior",
        "media_historica_municipio",
    ]

    numeric_features = calendar_features + accident_lag_features
    categorical_features = ["municipio"]

    return numeric_features, categorical_features


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", make_one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    estimator = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessamento", preprocessor),
            ("modelo", estimator),
        ]
    )


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.clip(np.asarray(y_pred), 0, None)
    nonzero = y_true_array != 0

    return {
        "MAE": float(mean_absolute_error(y_true_array, y_pred_array)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "MAPE_%_sem_zeros": (
            float(np.mean(np.abs((y_true_array[nonzero] - y_pred_array[nonzero]) / y_true_array[nonzero])) * 100)
            if nonzero.any()
            else float("nan")
        ),
        "WAPE_%": (
            float(np.sum(np.abs(y_true_array - y_pred_array)) / np.sum(y_true_array) * 100)
            if np.sum(y_true_array) > 0
            else float("nan")
        ),
        "R2": float(r2_score(y_true_array, y_pred_array)),
    }


def temporal_backtest(data: pd.DataFrame, features: list[str], pipeline: Pipeline) -> pd.DataFrame:
    results = []

    for test_year in sorted(year for year in data["ano"].unique() if year >= data["ano"].min() + 1):
        train = data[data["ano"] < test_year]
        test = data[data["ano"] == test_year]

        if train.empty or test.empty:
            continue

        fold_model = clone(pipeline)
        fold_model.fit(train[features], train[TARGET])
        predictions = fold_model.predict(test[features])

        row = {
            "modelo": MODEL_NAME,
            "versao": MODEL_VERSION,
            "conjunto_atributos": FEATURE_SET_NAME,
            "ano_teste": int(test_year),
            "n_treino": int(len(train)),
            "n_teste": int(len(test)),
        }
        row.update(calculate_metrics(test[TARGET], predictions))
        results.append(row)

    return pd.DataFrame(results)


def train_and_export(root: Path, output_dir: Path, cutoff_month: pd.Timestamp | None) -> tuple[Path, Path, pd.DataFrame]:
    if cutoff_month is None:
        cutoff_month = detect_last_complete_month(root)

    data = load_snapshot(root, cutoff_month=cutoff_month)
    data = add_temporal_features(data)
    numeric_features, categorical_features = feature_columns(data)
    features = numeric_features + categorical_features

    pipeline = make_pipeline(numeric_features, categorical_features)
    backtest_results = temporal_backtest(data, features, pipeline)

    pipeline.fit(data[features], data[TARGET])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"modelo_acidentes_rf_baseline_{MODEL_VERSION}.joblib"
    metadata_path = output_dir / f"modelo_acidentes_rf_baseline_{MODEL_VERSION}_metadata.json"

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_set_name": FEATURE_SET_NAME,
        "target": TARGET,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(data)),
        "training_start_month": data["data_referencia"].min().date().isoformat(),
        "training_end_month": data["data_referencia"].max().date().isoformat(),
        "cutoff_month": cutoff_month.date().isoformat(),
        "municipios": int(data["municipio"].nunique()),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_current_month_columns": LEAKAGE_CURRENT_MONTH_COLUMNS,
        "backtest_metrics_by_year": backtest_results.to_dict(orient="records"),
        "backtest_metrics_mean": (
            backtest_results[["MAE", "RMSE", "MAPE_%_sem_zeros", "WAPE_%", "R2"]]
            .mean()
            .to_dict()
            if not backtest_results.empty
            else {}
        ),
        "training_note": (
            "Modelo final treinado somente com meses completos do snapshot. "
            "Atributos do mes atual relacionados a mortos, feridos e veiculos foram "
            "excluidos para reduzir vazamento temporal."
        ),
    }

    payload = {
        "model": pipeline,
        "metadata": metadata,
        "feature_columns": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET,
    }

    joblib.dump(payload, model_path, compress=3)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return model_path, metadata_path, backtest_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina e exporta o modelo baseline de acidentes.")
    parser.add_argument(
        "--cutoff-month",
        type=str,
        default=None,
        help="Mes final de treino no formato YYYY-MM. Por padrao usa o ultimo mes observado nos dados intermediarios.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretorio de saida dos artefatos. Padrao: models/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    output_dir = args.output_dir or root / "models"
    cutoff_month = pd.Timestamp(f"{args.cutoff_month}-01") if args.cutoff_month else None

    model_path, metadata_path, backtest_results = train_and_export(root, output_dir, cutoff_month)

    print(f"Modelo exportado em: {model_path.relative_to(root)}")
    print(f"Metadados exportados em: {metadata_path.relative_to(root)}")

    if not backtest_results.empty:
        print("\nMetricas de backtesting por ano:")
        print(backtest_results.to_string(index=False))
        print("\nMetricas medias:")
        print(backtest_results[["MAE", "RMSE", "MAPE_%_sem_zeros", "WAPE_%", "R2"]].mean().to_string())


if __name__ == "__main__":
    main()
