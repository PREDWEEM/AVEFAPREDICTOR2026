
# ===============================================================
# 🌾 PREDWEEM v8 — Clasificador METEO → EMERREL (ANN) → PATRÓN
# Autor: Guillermo Chantre + ChatGPT
# Integración total del pipeline 2024–2025
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from predweem_ann_loader import ANN
from predweem_predictor import predecir_patron
import joblib

# ----------------------
# CONFIGURACIÓN STREAMLIT
# ----------------------
st.set_page_config(
    page_title="PREDWEEM v8 — Clasificación de Patrones",
    layout="wide"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------
# CARGA DEL MODELO ANN + CLASIFICADOR FINAL
# ----------------------
@st.cache_resource
def load_ann():
    return ANN()

@st.cache_resource
def load_classifier():
    model_path = "/mnt/data/predweem_meteo2patron.pkl"
    return joblib.load(model_path)

ann = load_ann()
clf = load_classifier()

# ----------------------
# SIDEBAR
# ----------------------
st.sidebar.title("Opciones")

input_file = st.sidebar.file_uploader(
    "Subir archivo meteorológico (CSV)",
    type=["csv"],
    help="Debe contener: JD, TMIN, TMAX, Prec"
)

rango = st.sidebar.radio(
    "Rango para graficar",
    ["Todo el año", "1/feb → 1/nov"]
)

st.sidebar.write("---")
ALPHA = st.sidebar.slider("Opacidad relleno EMERREL MA5", 0.0, 1.0, 0.6, 0.05)


# ----------------------
# FUNCIÓN SANITIZADORA BÁSICA
# ----------------------
def sanitize(df):
    df.columns = [c.strip().lower() for c in df.columns]

    mapping = {
        "jd": "JD",
        "julian_days": "JD",
        "dia_juliano": "JD",
        "tmin": "TMIN",
        "tmax": "TMAX",
        "prec": "Prec",
        "lluvia": "Prec"
    }

    new = {}
    for c in df.columns:
        if c in mapping:
            new[c] = mapping[c]
        else:
            new[c] = c

    df = df.rename(columns=new)
    needed = ["JD","TMIN","TMAX","Prec"]
    for n in needed:
        if n not in df.columns:
            st.error(f"Falta la columna obligatoria: {n}")
            st.stop()

    df = df[["JD","TMIN","TMAX","Prec"]].dropna()
    return df


# ----------------------
# FUNCIÓN DE CÁLCULO DE PERCENTILES
# ----------------------
def percentile_day(jd, emeac, p):
    if emeac.max() < p:
        return np.nan
    return np.interp(p, emeac, jd)


# ----------------------
# INTERFAZ PRINCIPAL
# ----------------------
st.title("🌾 PREDWEEM v8 — Clasificación Completa de Patrones de Emergencia")

st.markdown("""
Carga un archivo **meteorológico diario**, se ejecuta la **ANN PREDWEEM**, se genera
**EMERREL → EMEAC → JD25–JD95**, y finalmente se clasifica el patrón.
""")

if input_file is None:
    st.info("📄 Subí un archivo en el panel izquierdo para comenzar.")
    st.stop()

# ----------------------
# PROCESAMIENTO DEL ARCHIVO
# ----------------------
df = pd.read_csv(input_file)
df = sanitize(df)

st.success("Archivo cargado correctamente.")

# ORDENAR POR JD
df = df.sort_values("JD").reset_index(drop=True)

# ----------------------
# PREDICCIÓN ANN
# ----------------------
emerrel, emeac = ann.predict_emerrel(df)

df["EMERREL"] = emerrel
df["EMEAC"] = emeac
df["MA5"] = df["EMERREL"].rolling(5, min_periods=1).mean()

# ----------------------
# PERCENTILES
# ----------------------
JD = df["JD"].values
JD25 = percentile_day(JD, emeac, 0.25)
JD50 = percentile_day(JD, emeac, 0.50)
JD75 = percentile_day(JD, emeac, 0.75)
JD95 = percentile_day(JD, emeac, 0.95)

# ----------------------
# CLASIFICACIÓN FINAL METEO → PATRÓN
# ----------------------
# Usamos el dataset meteorológico para extraer features
def extraer_features(df):
    df2 = df.copy()
    df2["Tmed"] = (df2["TMIN"] + df2["TMAX"]) / 2
    feats = {}
    feats["Tmin_mean"] = df2["TMIN"].mean()
    feats["Tmax_mean"] = df2["TMAX"].mean()
    feats["Tmed_mean"] = df2["Tmed"].mean()
    feats["Prec_total"] = df2["Prec"].sum()
    feats["Prec_days_10mm"] = (df2["Prec"]>=10).sum()
    feats["Tmed_FM"] = df2[df2["JD"]<=121]["Tmed"].mean()
    feats["Prec_FM"] = df2[df2["JD"]<=121]["Prec"].sum()
    return feats

X = pd.DataFrame([extraer_features(df)])
proba = clf.predict_proba(X)[0]
clases = clf.classes_
PATRON = clases[np.argmax(proba)]
PROB_DICT = {c: float(p) for c,p in zip(clases, proba)}

# ----------------------
# MOSTRAR RESULTADOS
# ----------------------
st.header("📊 Resultado de la Clasificación")

col1, col2 = st.columns(2)

with col1:
    st.success(f"### 🏷️ Patrón predicho: **{PATRON}**")
    st.write("Probabilidades:")
    st.json(PROB_DICT)

with col2:
    st.info("### 📌 Percentiles (ANN → EMEAC)")
    st.write(f"- **JD25%** = {JD25:.1f}")
    st.write(f"- **JD50%** = {JD50:.1f}")
    st.write(f"- **JD75%** = {JD75:.1f}")
    st.write(f"- **JD95%** = {JD95:.1f}")

# ----------------------
# GRÁFICO EMERREL
# ----------------------
st.subheader("🌱 EMERGENCIA RELATIVA (ANN)")

fig_er = go.Figure()
fig_er.add_bar(
    x=df["JD"], y=df["EMERREL"],
    marker=dict(color="cornflowerblue"),
    name="EMERREL"
)

fig_er.add_trace(go.Scatter(
    x=df["JD"], y=df["MA5"],
    mode="lines",
    line=dict(color="black", width=3),
    name="MA5"
))

fig_er.update_layout(
    height=500,
    xaxis_title="Día Juliano",
    yaxis_title="EMERREL (0-1)",
    hovermode="x unified"
)

st.plotly_chart(fig_er, use_container_width=True)

# ----------------------
# GRÁFICO EMEAC
# ----------------------
st.subheader("📈 EMERGENCIA ACUMULADA (ANN)")

fig_ac = go.Figure()
fig_ac.add_trace(go.Scatter(
    x=df["JD"], y=df["EMEAC"],
    mode="lines",
    line=dict(color="green", width=3),
    name="EMEAC"
))

# Líneas de percentiles
for v, name in [(JD25, "25%"), (JD50, "50%"), (JD75, "75%"), (JD95, "95%")]:
    fig_ac.add_vline(x=v, line_dash="dot", annotation_text=name, annotation_position="top")

fig_ac.update_layout(
    height=500,
    xaxis_title="Día Juliano",
    yaxis_title="EMEAC (0-1)",
    hovermode="x unified"
)

st.plotly_chart(fig_ac, use_container_width=True)

# ----------------------
# DESCARGA DE RESULTADOS
# ----------------------
st.subheader("📥 Descargar resultados")

res = df[["JD","TMIN","TMAX","Prec","EMERREL","EMEAC","MA5"]].copy()
csv = res.to_csv(index=False)

st.download_button(
    "Descargar tabla EMERREL/EMEAC (CSV)",
    data=csv,
    file_name="PREDWEEM_v8_resultados.csv",
    mime="text/csv"
)

st.success("Pipeline PREDWEEM v8 completado con éxito.")
