"""MVP de previsão mensal de acidentes nas rodovias federais de SC."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from src.inference import load_model, load_history, predict, MODEL_PATH, METADATA_PATH, DATA_PATH
from src.prediction_features import available_months, build_features

DIAGNOSTICS_DIR = Path(__file__).resolve().parent / "data" / "processed"

st.set_page_config(page_title="Acidentes em SC | Previsão mensal", page_icon="🚦", layout="wide")


@st.cache_resource
def cached_model(file_signature):
    return load_model()


@st.cache_data
def cached_history(metadata, file_signature):
    return load_history(metadata)


@st.cache_data
def cached_diagnostics(file_signature):
    return tuple(pd.read_csv(DIAGNOSTICS_DIR / name) for name in (
        "ranking_baseline_modelos.csv", "erros_rf_por_municipio.csv", "erros_rf_por_faixa.csv"
    ))


def signature(*paths):
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


st.title("Previsão mensal de acidentes em Santa Catarina")
st.write("Consulte a quantidade estimada de acidentes nas rodovias federais de um município, com base no histórico da PRF.")
try:
    # Argumentos sem prefixo '_' para invalidar o cache quando os arquivos mudam.
    payload = cached_model(signature(MODEL_PATH, METADATA_PATH))
    metadata = payload["metadata"]
    history = cached_history(metadata, signature(DATA_PATH))
except Exception as exc:
    st.error(f"Não foi possível carregar o modelo e a base histórica: {exc}")
    st.info("Confira os arquivos em models/ e data/processed/ e as dependências do README.")
    st.stop()

months = available_months(history)
cutoff = history.data_referencia.max()
st.caption(f"Histórico disponível até {cutoff:%m/%Y} · Modelo {metadata['model_version']} · Dados mensais por município")
st.info("A previsão futura disponível é o mês seguinte ao último mês da base. Consultas históricas usam o modelo final, que já foi treinado nesses períodos; não representam uma avaliação independente.")

with st.form("prediction"):
    left, right = st.columns(2)
    municipality = left.selectbox("Município", sorted(history.municipio.unique()))
    month = right.selectbox("Mês de referência", months, index=len(months)-1, format_func=lambda x: x.strftime("%m/%Y"))
    submitted = st.form_submit_button("Gerar previsão", type="primary")

if submitted:
    try:
        features = build_features(history, municipality, month, metadata)
        result = predict(payload, features)
        st.session_state["result"] = (municipality, month, result)
    except ValueError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))

if "result" in st.session_state:
    city, date, estimate = st.session_state["result"]
    st.subheader(f"{city} · {date:%m/%Y}")
    st.metric("Quantidade estimada de acidentes no mês", f"{estimate:.1f}".replace(".", ","))
    st.caption("Estimativa aproximada; o valor decimal representa a saída do modelo de regressão.")
    observed = history.loc[(history.municipio == city) & (history.data_referencia <= date)].tail(24)
    chart_data = observed.rename(columns={"data_referencia": "Mês", "qtd_acidentes": "Acidentes"})[["Mês", "Acidentes"]].copy()
    chart_data["Série"] = "Histórico observado"
    forecast = pd.DataFrame({"Mês": [date], "Acidentes": [estimate], "Série": ["Estimativa do modelo"]})
    base = alt.Chart(pd.concat([chart_data, forecast])).encode(
        x=alt.X("Mês:T", axis=alt.Axis(format="%m/%Y")), y=alt.Y("Acidentes:Q", scale=alt.Scale(zero=True)),
        color=alt.Color("Série:N", scale=alt.Scale(domain=["Histórico observado", "Estimativa do modelo"], range=["#2878B5", "#E76F51"])),
        tooltip=[alt.Tooltip("Mês:T", format="%m/%Y"), "Série:N", alt.Tooltip("Acidentes:Q", format=".1f")])
    st.altair_chart(base.transform_filter(alt.datum.Série == "Histórico observado").mark_line(point=True) + base.transform_filter(alt.datum.Série == "Estimativa do modelo").mark_point(size=160, filled=True), width="stretch")
    st.download_button("Baixar resultado (CSV)", pd.DataFrame([{"municipio": city, "mes": date.strftime("%Y-%m"), "previsao_acidentes": estimate, "versao_modelo": metadata["model_version"]}]).to_csv(index=False).encode("utf-8-sig"), "previsao_acidentes.csv", "text/csv")
else:
    st.caption("Selecione o município e o mês e clique em Gerar previsão para visualizar o resultado.")

with st.expander("Sobre o modelo e sua validação"):
    st.write(f"**Modelo:** {metadata['model_name']} · **Versão:** {metadata['model_version']}")
    st.write(f"Treinamento de {pd.Timestamp(metadata['training_start_month']):%m/%Y} a {pd.Timestamp(metadata['training_end_month']):%m/%Y}, com {metadata['municipios']} municípios.")
    metrics = metadata["backtest_metrics_mean"]
    a, b, c = st.columns(3)
    a.metric("MAE médio (acidentes)", f"{metrics['MAE']:.2f}")
    b.metric("RMSE médio (acidentes)", f"{metrics['RMSE']:.2f}")
    c.metric("WAPE médio", f"{metrics['WAPE_%']:.1f}%")
    st.caption("Médias dos backtests temporais por ano, registradas na Sprint anterior. Não são intervalos de confiança da consulta atual.")
    st.dataframe(pd.DataFrame(metadata["backtest_metrics_by_year"])[["ano_teste", "n_teste", "MAE", "RMSE", "WAPE_%", "R2"]], hide_index=True, width="stretch")

with st.expander("Comparação e erros dos backtests"):
    diagnostic_paths = [DIAGNOSTICS_DIR / name for name in (
        "ranking_baseline_modelos.csv", "erros_rf_por_municipio.csv", "erros_rf_por_faixa.csv"
    )]
    try:
        ranking, city_errors, band_errors = cached_diagnostics(signature(*diagnostic_paths))
        rf = ranking.loc[(ranking.modelo == "RandomForest") & (ranking.conjunto_atributos == metadata["feature_set_name"])].iloc[0]
        naive = ranking.loc[ranking.modelo == "NaiveUltimoMes"].iloc[0]
        improvement = 100 * (naive.MAE_medio - rf.MAE_medio) / naive.MAE_medio
        st.write(f"O Random Forest reduziu o MAE médio em **{improvement:.1f}%** em relação à repetição do último mês, nas quatro janelas avaliadas.")
        comparison = ranking.loc[(ranking.conjunto_atributos == metadata["feature_set_name"]) | (ranking.modelo == "NaiveUltimoMes")]
        st.dataframe(comparison[["modelo", "MAE_medio", "RMSE_medio", "WAPE_medio"]].rename(columns={
            "modelo": "Modelo", "MAE_medio": "MAE médio", "RMSE_medio": "RMSE médio", "WAPE_medio": "WAPE médio (%)"
        }), hide_index=True, width="stretch")
        st.caption("Médias simples por janela: testes em 2023, 2024, 2025 e janeiro a maio de 2026. Os algoritmos supervisionados usam calendário, histórico de sinistros e município.")
        diagnostic_city = st.selectbox("Município para consultar os erros de teste", sorted(city_errors.municipio), key="diagnostic_city")
        city_row = city_errors.loc[city_errors.municipio == diagnostic_city]
        labels = {"n": "Meses de teste", "total_observado": "Total observado", "MAE": "MAE", "RMSE": "RMSE", "WAPE_%": "WAPE (%)", "vies_medio": "Viés médio"}
        st.dataframe(city_row[list(labels)].rename(columns=labels), hide_index=True, width="stretch")
        st.write("**Erros por quantidade observada no município-mês**")
        st.dataframe(band_errors.rename(columns={"faixa_observada": "Sinistros observados", **labels}), hide_index=True, width="stretch")
        st.caption("As tabelas de erros reúnem as 5.207 previsões dos backtests. Viés = previsto menos observado; valores negativos indicam subestimação. WAPE é indefinido quando o total observado é zero. As faixas usam a contagem conhecida após o teste e não são categorias de previsão.")
        st.download_button("Baixar erros por município (CSV)", city_errors.to_csv(index=False).encode("utf-8-sig"), "erros_backtests_municipios.csv", "text/csv")
    except (OSError, ValueError, KeyError, IndexError):
        st.warning("A análise detalhada dos backtests está indisponível nesta versão da aplicação.")
