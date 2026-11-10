"""Gera tabelas e figuras dos backtests salvos, sem treinar modelos."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pgp-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed"
FIGURES = ROOT / "docs/graphics"


def summarize(frame):
    error = frame.y_pred - frame.y_real
    total = frame.y_real.sum()
    return {
        "n": len(frame), "total_observado": total,
        "MAE": error.abs().mean(), "RMSE": np.sqrt((error ** 2).mean()),
        "WAPE_%": 100 * error.abs().sum() / total if total else np.nan,
        "vies_medio": error.mean(),
    }


def main():
    predictions = pd.read_csv(OUTPUT / "predicoes_baseline_modelos.csv", parse_dates=["data_referencia"])
    selected = predictions.loc[
        predictions.modelo.eq("RandomForest")
        & predictions.conjunto_atributos.eq("calendario_lags_municipio")
    ].copy()
    keys = ["municipio", "data_referencia"]
    if selected.duplicated(keys).any() or len(selected) != 5207:
        raise ValueError("Backtests inesperados: conferir duplicidades e cobertura temporal.")
    results = pd.read_csv(OUTPUT / "resultados_baseline_modelos.csv")
    for year, group in selected.groupby("ano_teste"):
        saved = results.loc[results.modelo.eq("RandomForest")
                            & results.conjunto_atributos.eq("calendario_lags_municipio")
                            & results.ano_teste.eq(year)].iloc[0]
        for metric in ["MAE", "RMSE", "WAPE_%"]:
            np.testing.assert_allclose(summarize(group)[metric], saved[metric], rtol=1e-10)

    city_rows = [{"municipio": city, **summarize(group)} for city, group in selected.groupby("municipio")]
    cities = pd.DataFrame(city_rows).sort_values(["MAE", "municipio"], ascending=[False, True])
    cities.to_csv(OUTPUT / "erros_rf_por_municipio.csv", index=False)
    selected["faixa_observada"] = pd.cut(selected.y_real, [-1, 0, 5, 10, 20, np.inf],
                                         labels=["0", "1-5", "6-10", "11-20", "21 ou mais"])
    bands = pd.DataFrame([
        {"faixa_observada": str(band), **summarize(group)}
        for band, group in selected.groupby("faixa_observada", observed=True, sort=True)
    ])
    bands.to_csv(OUTPUT / "erros_rf_por_faixa.csv", index=False)
    selected.groupby("data_referencia")[["y_real", "y_pred"]].sum().to_csv(OUTPUT / "backtest_rf_totais_mensais.csv")

    # Seleção pelo volume observado, sem escolher municípios pelo bom ajuste.
    largest = cities.nlargest(3, "total_observado").municipio.tolist()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout="constrained")
    for ax, city in zip(axes, largest):
        rows = selected.loc[selected.municipio.eq(city)].sort_values("data_referencia")
        ax.plot(rows.data_referencia, rows.y_real, label="Observado", color="#245a81")
        ax.plot(rows.data_referencia, rows.y_pred, label="Previsto", color="#b34e24", linestyle="--")
        ax.set_title(city.title(), loc="left", fontsize=11)
        ax.set_ylabel("Sinistros/mês")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    axes[0].legend(ncol=2, frameon=False)
    axes[-1].set_xlabel("Mês de teste (2023 a maio/2026)")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "backtest_rf_municipios.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.3), layout="constrained")
    ax.bar(bands.faixa_observada, bands.MAE, color="#245a81")
    ax.set_xlabel("Quantidade observada de sinistros no município-mês")
    ax.set_ylabel("MAE (sinistros)")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(FIGURES / "backtest_rf_erros_faixa.png", dpi=180)
    plt.close(fig)
    print(bands.to_string(index=False))
    print(cities.head(5).to_string(index=False))
    print(f"Municípios com total zero no teste: {(cities.total_observado == 0).sum()}")


if __name__ == "__main__":
    main()
