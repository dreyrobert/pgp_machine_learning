"""Entradas mensais reproduzindo as regras do baseline v1."""
import numpy as np
import pandas as pd


def available_months(history):
    start = history.data_referencia.min() + pd.DateOffset(months=12)
    end = history.data_referencia.max() + pd.DateOffset(months=1)
    return list(pd.date_range(start, end, freq="MS"))


def build_features(history, municipality, month, metadata):
    month = pd.Timestamp(month).to_period("M").to_timestamp()
    if municipality not in history.municipio.unique():
        raise ValueError("Município indisponível na base histórica.")
    if month not in available_months(history):
        raise ValueError("Mês indisponível: escolha um mês com histórico suficiente ou o próximo mês da base.")
    previous = history.loc[(history.municipio == municipality) & (history.data_referencia < month)].sort_values("data_referencia")
    expected = pd.date_range(history.data_referencia.min(), month - pd.DateOffset(months=1), freq="MS")
    if list(previous.data_referencia) != list(expected) or len(previous) < 12:
        raise ValueError("Histórico mensal insuficiente ou com meses ausentes para este município.")
    values = previous.qtd_acidentes.astype(float)
    row = {
        "municipio": municipality, "ano": month.year, "mes": month.month,
        "trimestre": month.quarter, "eh_alta_temporada": int(month.month in [12, 1, 2, 7]),
        "tempo_idx": (month.year - pd.Timestamp(metadata["training_start_month"]).year) * 12 + month.month,
        "mes_sin": np.sin(2 * np.pi * month.month / 12),
        "mes_cos": np.cos(2 * np.pi * month.month / 12),
        "lag_acidentes_1m": values.iloc[-1], "lag_acidentes_2m": values.iloc[-2],
        "lag_acidentes_12m": values.iloc[-12], "media_movel_3m": values.tail(3).mean(),
        "media_movel_6m": values.tail(6).mean(),
        "acidentes_acum_ano_ate_mes_anterior": previous.loc[previous.data_referencia.dt.year == month.year, "qtd_acidentes"].sum(),
        "media_historica_municipio": values.mean(),
    }
    return pd.DataFrame([row])[metadata["numeric_features"] + metadata["categorical_features"]]
