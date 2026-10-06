"""MVP de previsão mensal de acidentes nas rodovias federais de SC."""
import altair as alt
import pandas as pd
import streamlit as st
from src.inference import load_model, load_history, predict, MODEL_PATH, METADATA_PATH, DATA_PATH
from src.prediction_features import available_months, build_features

st.set_page_config(page_title="Acidentes em SC | Previsão mensal", page_icon="🚦", layout="wide")


@st.cache_resource
def cached_model(file_signature):
    return load_model()


@st.cache_data
def cached_history(metadata, file_signature):
    return load_history(metadata)


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
