# app_predweem_v8.py
# ===============================================================
# 🌾 PREDWEEM v8 — METEO → ANN (EMERREL/EMEAC) → PATRÓN
# Usa:
#   - predweem_ann_loader.ANN  (IW, LW, bias_*)
#   - predweem_meteo2patron.pkl (clasificador METEO→PATRÓN)
#   - predweem_predictor.predecir_patron
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from predweem_ann_loader import ANN
from predweem_predictor import predecir_patron

# ---------------- CONFIG STREAMLIT ----------------
st.set_page_config(
    page_title="PREDWEEM v8 — Clasificación de patrones",
    layout="wide"
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌾 PREDWEEM v8 — Clasificación de patrones de emergencia (Lolium / AVEFA)")

st.markdown(
    """
    Esta app:
    1. Lee un archivo de **meteorología diaria** (JD, TMIN, TMAX, Prec).  
    2. Usa la **ANN** para generar la curva de **EMERREL** y **EMEAC**.  
    3. Calcula los **percentiles JD25–JD95**.  
    4. Aplica el modelo **METEO → PATRÓN** y devuelve la clasificación.
    """
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Opciones de entrada")

uploaded_file = st.sidebar.file_uploader(
    "Subir archivo meteorológico (CSV)",
    type=["csv"],
    help="Debe contener al menos: JD, TMIN, TMAX, Prec (nombres pueden estar en minúsculas)."
)

rango_plot = st.sidebar.radio(
    "Rango de visualización",
    ["Todo el año (según JD)", "1/feb → 1/nov"],
    index=0
)

ALPHA = st.sidebar.slider(
    "Opacidad relleno MA5 EMERREL",
    min_value=0.0, max_value=1.0,
    value=0.6, step=0.05
)

# ---------------- FUNCIONES AUX ----------------
def sanitize_meteo(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y deja JD,TMIN,TMAX,Prec."""
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    mapping = {
        "jd": "JD",
        "julian_days": "JD",
        "dia_juliano": "JD",
        "tmin": "TMIN",
        "tmax": "TMAX",
        "temp_min": "TMIN",
        "temp_max": "TMAX",
        "prec": "Prec",
        "lluvia": "Prec",
        "ppt": "Prec"
    }

    new_cols = {}
    for c in df.columns:
        if c in mapping:
            new_cols[c] = mapping[c]
        else:
            new_cols[c] = c

    df = df.rename(columns=new_cols)

    required = ["JD", "TMIN", "TMAX", "Prec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas obligatorias: {missing}")
        st.stop()

    df = df[["JD", "TMIN", "TMAX", "Prec"]].copy()
    for c in ["JD", "TMIN", "TMAX", "Prec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("JD").reset_index(drop=True)

    return df

def calc_percentile_day(jd, emeac, p):
    """Devuelve JD donde EMEAC cruza el percentil p (0–1)."""
    jd = np.asarray(jd, float)
    emeac = np.asarray(emeac, float)
    if emeac.max() < p:
        return np.nan
    return np.interp(p, emeac, jd)

# ---------------- CARGA ANN ----------------
@st.cache_resource
def load_ann():
    return ANN()

ann = load_ann()

# ---------------- FLUJO PRINCIPAL ----------------
if uploaded_file is None:
    st.info("📄 Subí un archivo de meteorología diaria en la barra lateral para comenzar.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file)
except Exception:
    st.error("No pude leer el archivo como CSV. Verificá el formato.")
    st.stop()

df_meteo = sanitize_meteo(df_raw)
st.success("✅ Archivo meteorológico cargado y saneado correctamente.")

# --------- PREDICCIÓN ANN ---------
emerrel, emeac = ann.predict_emerrel(df_meteo)
df_meteo["EMERREL"] = emerrel
df_meteo["EMEAC"] = emeac
df_meteo["MA5"] = df_meteo["EMERREL"].rolling(5, min_periods=1).mean()

JD = df_meteo["JD"].to_numpy()
JD25 = calc_percentile_day(JD, emeac, 0.25)
JD50 = calc_percentile_day(JD, emeac, 0.50)
JD75 = calc_percentile_day(JD, emeac, 0.75)
JD95 = calc_percentile_day(JD, emeac, 0.95)

# --------- CLASIFICACIÓN METEO → PATRÓN ---------
res_patron = predecir_patron(df_meteo)
PATRON = res_patron["clasificacion"]
PROBAS = res_patron["probabilidades"]

# ---------------- SALIDA PRINCIPAL ----------------
st.header("📊 Resultado de la clasificación")

col1, col2 = st.columns(2)

with col1:
    st.success(f"### 🏷️ Patrón predicho: **{PATRON}**")
    st.write("Probabilidades por patrón:")
    st.json(PROBAS)

with col2:
    st.info("### 📌 Percentiles de emergencia acumulada (ANN → EMEAC)")
    st.write(f"- **JD25%** = {JD25:.1f}" if not np.isnan(JD25) else "- JD25% = NA")
    st.write(f"- **JD50%** = {JD50:.1f}" if not np.isnan(JD50) else "- JD50% = NA")
    st.write(f"- **JD75%** = {JD75:.1f}" if not np.isnan(JD75) else "- JD75% = NA")
    st.write(f"- **JD95%** = {JD95:.1f}" if not np.isnan(JD95) else "- JD95% = NA")

# ---------------- GRÁFICO EMERREL ----------------
st.subheader("🌱 EMERGENCIA RELATIVA (ANN)")

df_plot = df_meteo.copy()
if rango_plot == "1/feb → 1/nov":
    mask = (df_plot["JD"] >= 32) & (df_plot["JD"] <= 305)
    if mask.any():
        df_plot = df_plot[mask].copy()

fig_er = go.Figure()
fig_er.add_bar(
    x=df_plot["JD"],
    y=df_plot["EMERREL"],
    marker=dict(color="cornflowerblue"),
    name="EMERREL"
)

fig_er.add_trace(go.Scatter(
    x=df_plot["JD"],
    y=df_plot["MA5"],
    mode="lines",
    line=dict(color="black", width=3),
    name="Media móvil 5 días"
))

fig_er.update_layout(
    xaxis_title="Día Juliano",
    yaxis_title="EMERREL (0–1)",
    hovermode="x unified",
    height=450
)

st.plotly_chart(fig_er, use_container_width=True)

# ---------------- GRÁFICO EMEAC ----------------
st.subheader("📈 EMERGENCIA ACUMULADA (ANN)")

fig_ac = go.Figure()
fig_ac.add_trace(go.Scatter(
    x=df_plot["JD"],
    y=df_plot["EMEAC"],
    mode="lines",
    line=dict(color="green", width=3),
    name="EMEAC"
))

for v, label in [(JD25, "25%"), (JD50, "50%"), (JD75, "75%"), (JD95, "95%")]:
    if not np.isnan(v):
        fig_ac.add_vline(
            x=v,
            line_dash="dot",
            annotation_text=label,
            annotation_position="top"
        )

fig_ac.update_layout(
    xaxis_title="Día Juliano",
    yaxis_title="EMEAC (0–1)",
    hovermode="x unified",
    height=450
)

st.plotly_chart(fig_ac, use_container_width=True)

# ---------------- TABLA Y DESCARGA ----------------
st.subheader("📥 Tabla de resultados")

tabla = df_meteo[["JD", "TMIN", "TMAX", "Prec", "EMERREL", "EMEAC", "MA5"]].copy()
st.dataframe(tabla, use_container_width=True)

csv = tabla.to_csv(index=False)
st.download_button(
    "Descargar resultados (CSV)",
    data=csv,
    file_name="PREDWEEM_v8_EMERREL_EMEAC.csv",
    mime="text/csv"
)

st.success("✅ PREDWEEM v8 ejecutado correctamente sobre el archivo subido.")
