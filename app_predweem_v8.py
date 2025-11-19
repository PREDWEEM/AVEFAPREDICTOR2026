# app_predweem_v8.py
# ===============================================================
# 🌾 PREDWEEM v8 — METEO → ANN (EMERREL/EMEAC) → PATRÓN
# Versión corregida y adaptada al archivo real:
# Bordenave_1977_2015_por_anio_con_JD.xlsx
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ---------------------------------------------------------------
# VISUAL CONFIG STREAMLIT
# ---------------------------------------------------------------
st.set_page_config(page_title="PREDWEEM v8 — Clasificación", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 PREDWEEM v8 — Clasificación de patrones de emergencia")

# ===============================================================
# ANN LOADER
# ===============================================================
class ANN:
    def __init__(self):
        self.IW  = np.load("IW.npy")
        self.bIW = np.load("bias_IW.npy")
        self.LW  = np.load("LW.npy")
        self.bO  = np.load("bias_out.npy")

        if self.LW.ndim == 1:
            self.LW = self.LW.reshape(1, -1)

        self.bO = float(self.bO if np.ndim(self.bO)==0 else np.ravel(self.bO)[0])

        self.input_min = np.array([1, -7, 0, 0], dtype=float)
        self.input_max = np.array([300, 25.5, 41, 84], dtype=float)
        self._den = np.maximum(self.input_max - self.input_min, 1e-9)

    def _tansig(self, x): return np.tanh(x)

    def _normalize(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / self._den - 1

    def predict_emerrel(self, df):
        X = df[["JD", "TMIN", "TMAX", "Prec"]].to_numpy(float)
        Xn = self._normalize(X)

        z1 = Xn @ self.IW + self.bIW
        a1 = self._tansig(z1)

        z2 = (a1 @ self.LW.T).ravel() + self.bO
        emerrel = (self._tansig(z2) + 1) / 2
        emerrel = np.clip(emerrel, 0, 1)

        emeac = emerrel.cumsum()
        if emeac.max() > 0:
            emeac = emeac / emeac.max()

        return emerrel, emeac


@st.cache_resource
def load_ann():
    return ANN()

ann = load_ann()

# ===============================================================
# CLASIFICADOR METEO → PATRÓN
# ===============================================================
MODEL_PATH = "predweem_meteo2patron.pkl"

def extraer_features(df):
    df2 = df.copy()
    df2["Tmed"] = (df2["TMIN"] + df2["TMAX"]) / 2

    return {
        "Tmin_mean": df2["TMIN"].mean(),
        "Tmax_mean": df2["TMAX"].mean(),
        "Tmed_mean": df2["Tmed"].mean(),
        "Prec_total": df2["Prec"].sum(),
        "Prec_10mm": (df2["Prec"] >= 10).sum(),
        "Tmed_FM": df2[df2["JD"] <= 121]["Tmed"].mean(),
        "Prec_FM": df2[df2["JD"] <= 121]["Prec"].sum()
    }

@st.cache_resource
def load_clf():
    return joblib.load(MODEL_PATH)

def predecir_patron(df):
    X = pd.DataFrame([extraer_features(df)])
    model = load_clf()
    prob = model.predict_proba(X)[0]
    clases = model.classes_
    patron = clases[np.argmax(prob)]
    prob_dict = {c: float(p) for c, p in zip(clases, prob)}
    return {"clasificacion": patron, "probabilidades": prob_dict}

# ===============================================================
# LECTOR XLSX MULTI-AÑO — CORREGIDO
# ===============================================================
def cargar_multiples_anios_xlsx(uploaded):
    """
    Lee hojas con columnas:
    Fecha, JD, Temperatura_Minima, Temperatura_Maxima, Precipitacion_Pluviometrica
    """
    xls = pd.ExcelFile(uploaded)
    años = {}

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
        except:
            continue

        cols = {c.lower(): c for c in df.columns}

        col_jd   = cols.get("jd")
        col_tmin = cols.get("temperatura_minima")
        col_tmax = cols.get("temperatura_maxima")
        col_prec = cols.get("precipitacion_pluviometrica")

        if not (col_jd and col_tmin and col_tmax and col_prec):
            continue

        df2 = df[[col_jd, col_tmin, col_tmax, col_prec]].copy()
        df2.columns = ["JD", "TMIN", "TMAX", "Prec"]

        for c in ["JD", "TMIN", "TMAX", "Prec"]:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

        df2 = df2.dropna().sort_values("JD").reset_index(drop=True)

        try:
            año = int(sheet)
        except:
            continue

        años[año] = df2

    return años

# ===============================================================
# FUNCIONES AUXILIARES
# ===============================================================
def calc_percentile_day(jd, emeac, p):
    jd = np.asarray(jd, float)
    emeac = np.asarray(emeac, float)
    if emeac.max() < p:
        return np.nan
    return np.interp(p, emeac, jd)

# ===============================================================
# SIDEBAR — ENTRADA DE ARCHIVO
# ===============================================================
st.sidebar.header("Entrada de datos")

uploaded = st.sidebar.file_uploader(
    "Subir archivo meteorológico (CSV o XLSX multi-año)",
    type=["csv", "xlsx"]
)

if uploaded is None:
    st.info("Subí un archivo para comenzar…")
    st.stop()

# CSV → un solo año
if uploaded.name.lower().endswith(".csv"):
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    mapping = {
        "jd": "JD",
        "temperatura_minima": "TMIN",
        "tmin": "TMIN",
        "temperatura_maxima": "TMAX",
        "tmax": "TMAX",
        "precipitacion_pluviometrica": "Prec",
        "prec": "Prec"
    }
    df_raw = df_raw.rename(columns={c: mapping.get(c, c) for c in df_raw.columns})
    df_raw = df_raw[["JD", "TMIN", "TMAX", "Prec"]].dropna()
    años = {"CSV": df_raw}

# XLSX → múltiples años
else:
    años = cargar_multiples_anios_xlsx(uploaded)
    if len(años) == 0:
        st.error("No se encontraron hojas válidas en el XLSX.")
        st.stop()

st.success(f"Años detectados: {list(años.keys())}")

año_sel = st.sidebar.selectbox("Seleccionar año", sorted(años.keys()))
df_meteo = años[año_sel].copy()

# ===============================================================
# PREDICCIÓN ANN PARA EL AÑO — CORREGIDA (MA5 FIX)
# ===============================================================
emerrel, emeac = ann.predict_emerrel(df_meteo)

df_meteo["EMERREL"] = emerrel
df_meteo["EMEAC"] = emeac

# FIX: rolling requiere pandas.Series
df_meteo["MA5"] = (
    pd.Series(emerrel)
    .rolling(5, min_periods=1)
    .mean()
    .to_numpy()
)

JD = df_meteo["JD"].to_numpy()

JD25 = calc_percentile_day(JD, emeac, 0.25)
JD50 = calc_percentile_day(JD, emeac, 0.50)
JD75 = calc_percentile_day(JD, emeac, 0.75)
JD95 = calc_percentile_day(JD, emeac, 0.95)

# Clasificación patrón
res_pat = predecir_patron(df_meteo)

# ===============================================================
# MOSTRAR RESULTADOS
# ===============================================================
st.header(f"📊 Año {año_sel}")

col1, col2 = st.columns(2)

with col1:
    st.success(f"### 🏷️ Patrón predicho: **{res_pat['clasificacion']}**")
    st.json(res_pat["probabilidades"])

with col2:
    st.info("### Percentiles ANN/EMEAC")
    st.write(f"JD25 = {JD25:.1f}")
    st.write(f"JD50 = {JD50:.1f}")
    st.write(f"JD75 = {JD75:.1f}")
    st.write(f"JD95 = {JD95:.1f}")

# ===============================================================
# GRÁFICOS
# ===============================================================
st.subheader("🌱 EMERREL (ANN)")
fig = go.Figure()
fig.add_bar(x=df_meteo["JD"], y=df_meteo["EMERREL"], marker=dict(color="cornflowerblue"))
fig.add_trace(go.Scatter(x=df_meteo["JD"], y=df_meteo["MA5"], mode="lines",
                         line=dict(color="black", width=3)))
fig.update_layout(yaxis_title="EMERREL (0–1)", xaxis_title="JD")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 EMEAC (ANN)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_meteo["JD"], y=df_meteo["EMEAC"],
                          mode="lines", line=dict(color="green", width=3)))
for v, label in [(JD25, "25%"), (JD50, "50%"), (JD75, "75%"), (JD95, "95%")]:
    if not np.isnan(v):
        fig2.add_vline(x=v, line_dash="dot", annotation_text=label)
fig2.update_layout(yaxis_title="EMEAC (0–1)", xaxis_title="JD")
st.plotly_chart(fig2, use_container_width=True)

# ===============================================================
# TABLA Y DESCARGA
# ===============================================================
st.subheader("📥 Datos del año seleccionado")
st.dataframe(df_meteo, use_container_width=True)

st.download_button(
    "Descargar CSV del año",
    data=df_meteo.to_csv(index=False),
    file_name=f"PREDWEEM_v8_{año_sel}.csv",
    mime="text/csv"
)

# ===============================================================
# PROCESAR TODOS LOS AÑOS (opcional)
# ===============================================================
if len(años) > 1:
    st.header("📈 Procesar TODOS los años")

    if st.button("Ejecutar ANN + Clasificación para todos los años"):
        registros = []

        for a, dfy in años.items():
            emer_all, emac_all = ann.predict_emerrel(dfy)
            JDy = dfy["JD"].to_numpy()

            r = predecir_patron(dfy)

            registros.append({
                "Año": a,
                "Patrón": r["clasificacion"],
                "Prob_max": max(r["probabilidades"].values()),
                "JD25": calc_percentile_day(JDy, emac_all, 0.25),
                "JD50": calc_percentile_day(JDy, emac_all, 0.50),
                "JD75": calc_percentile_day(JDy, emac_all, 0.75),
                "JD95": calc_percentile_day(JDy, emac_all, 0.95)
            })

        tabla_all = pd.DataFrame(registros).sort_values("Año")
        st.dataframe(tabla_all, use_container_width=True)

        st.download_button(
            "Descargar CSV — todos los años",
            data=tabla_all.to_csv(index=False),
            file_name="PREDWEEM_v8_todos_los_años.csv",
            mime="text/csv"
        )

st.success("✔ PREDWEEM v8 ejecutado correctamente.")


