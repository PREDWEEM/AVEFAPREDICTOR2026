# app_predweem_v8.py
# ===============================================================
# 🌾 PREDWEEM v8 — METEO → ANN (EMERREL/EMEAC) → PATRÓN
# - Lee CSV (un año) o XLSX con múltiples hojas (una por año)
# - ANN (IW, LW, bias) → EMERREL diaria y EMEAC acumulada
# - Calcula percentiles JD25–JD95
# - Clasificador METEO → PATRÓN con predweem_meteo2patron.pkl
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ===============================================================
# CONFIGURACIÓN STREAMLIT
# ===============================================================
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
    Esta aplicación:
    1. Lee un archivo de **meteorología diaria** (CSV con un año o XLSX con varias hojas, una por año).  
    2. Usa la **red neuronal entrenada** para generar la curva de **EMERREL** y **EMEAC**.  
    3. Calcula los **percentiles JD25–JD95** de la emergencia acumulada.  
    4. Aplica el modelo **METEO → PATRÓN** para clasificar el año.
    """
)

# ===============================================================
# CLASE ANN (carga de pesos locales y predicción EMERREL/EMEAC)
# ===============================================================
class ANN:
    """
    Carga la red neuronal entrenada (IW, LW, bias_*) desde archivos .npy
    y permite predecir EMERREL y EMEAC a partir de JD, TMIN, TMAX, Prec.
    """

    def __init__(self):
        # Pesos y bias guardados en la carpeta de la app
        self.IW  = np.load("IW.npy")
        self.bIW = np.load("bias_IW.npy")
        self.LW  = np.load("LW.npy")
        self.bO  = np.load("bias_out.npy")

        if self.LW.ndim == 1:
            self.LW = self.LW.reshape(1, -1)

        # Aseguramos escalar a float
        self.bO = float(self.bO if np.ndim(self.bO) == 0 else np.ravel(self.bO)[0])

        # Mismos rangos de entrada que en tu app AVEFA
        # Orden: [JD, TMIN, TMAX, Prec]
        self.input_min = np.array([1, -7, 0, 0], dtype=float)
        self.input_max = np.array([300, 25.5, 41, 84], dtype=float)
        self._den = np.maximum(self.input_max - self.input_min, 1e-9)

    def _tansig(self, x):
        return np.tanh(x)

    def _normalize(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / self._den - 1

    def predict_emerrel(self, df_meteo: pd.DataFrame):
        """
        df_meteo debe tener columnas: JD, TMIN, TMAX, Prec.
        Devuelve:
        - emerrel: emergencia relativa diaria (0–1)
        - emeac: emergencia acumulada normalizada (0–1)
        """
        X = df_meteo[["JD", "TMIN", "TMAX", "Prec"]].to_numpy(float)
        Xn = self._normalize(X)

        z1 = Xn @ self.IW + self.bIW
        a1 = self._tansig(z1)

        z2 = (a1 @ self.LW.T).ravel() + self.bO
        y  = self._tansig(z2)          # [-1..1]
        emerrel = (y + 1) / 2          # → [0..1]
        emerrel = np.clip(emerrel, 0, 1)

        emeac = emerrel.cumsum()
        if emeac.max() > 0:
            emeac = emeac / emeac.max()

        return emerrel, emeac

# ===============================================================
# CLASIFICADOR METEO → PATRÓN (usa predweem_meteo2patron.pkl)
# ===============================================================
MODEL_PATH = "predweem_meteo2patron.pkl"

def extraer_features(df: pd.DataFrame):
    """
    Extrae las features climáticas usadas al entrenar el modelo METEO → PATRÓN.
    df debe tener columnas: JD, TMIN, TMAX, Prec.
    """
    df2 = df.copy()
    df2["Tmed"] = (df2["TMIN"] + df2["TMAX"]) / 2

    feats = {
        "Tmin_mean": df2["TMIN"].mean(),
        "Tmax_mean": df2["TMAX"].mean(),
        "Tmed_mean": df2["Tmed"].mean(),
        "Prec_total": df2["Prec"].sum(),
        "Prec_days_10mm": (df2["Prec"] >= 10).sum(),
        "Tmed_FM": df2[df2["JD"] <= 121]["Tmed"].mean(),
        "Prec_FM": df2[df2["JD"] <= 121]["Prec"].sum(),
    }
    return feats

@st.cache_resource
def load_clf():
    return joblib.load(MODEL_PATH)

def predecir_patron(df: pd.DataFrame):
    """
    Recibe un DataFrame con meteorología diaria (JD,TMIN,TMAX,Prec),
    calcula features y aplica el modelo predweem_meteo2patron.pkl.
    """
    # Normalizamos nombres de columnas posibles
    df = df.rename(columns={
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
    })

    df = df[["JD", "TMIN", "TMAX", "Prec"]].dropna()

    X = pd.DataFrame([extraer_features(df)])

    model = load_clf()
    probas = model.predict_proba(X)[0]
    clases = model.classes_

    patron = clases[probas.argmax()]
    prob_dict = {c: float(p) for c, p in zip(clases, probas)}

    return {
        "clasificacion": patron,
        "probabilidades": prob_dict
    }

# ===============================================================
# FUNCIONES AUXILIARES
# ===============================================================
def sanitize_meteo(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y deja JD,TMIN,TMAX,Prec para CSV de un solo año."""
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    mapping = {
        "jd": "JD",
        "julian_days": "JD",
        "dia_juliano": "JD",
        "tmin": "TMIN",
        "temperatura_minima": "TMIN",
        "tmax": "TMAX",
        "temperatura_maxima": "TMAX",
        "temp_min": "TMIN",
        "temp_max": "TMAX",
        "prec": "Prec",
        "precipitacion": "Prec",
        "lluvia": "Prec",
        "ppt": "Prec"
    }

    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})

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

def cargar_multiples_anios_xlsx(uploaded_file):
    """
    Lee un XLSX con múltiples hojas (una por año).
    Devuelve:
    - dict: {año: DataFrame}
    """
    xls = pd.ExcelFile(uploaded_file)
    años = {}

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue

        df.columns = [c.strip().lower() for c in df.columns]

        mapping = {
            "fecha": "Fecha",
            "jd": "JD",
            "julian_days": "JD",
            "dia_juliano": "JD",
            "tmin": "TMIN",
            "temperatura_minima": "TMIN",
            "tmax": "TMAX",
            "temperatura_maxima": "TMAX",
            "prec": "Prec",
            "precipitacion": "Prec",
            "lluvia": "Prec",
            "ppt": "Prec"
        }
        df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})

        # requerimos JD,TMIN,TMAX,Prec
        if not {"JD", "TMIN", "TMAX", "Prec"}.issubset(df.columns):
            continue

        df = df[["JD", "TMIN", "TMAX", "Prec"]].dropna()
        df = df.sort_values("JD").reset_index(drop=True)

        if df.shape[0] > 0:
            try:
                año = int(sheet)
            except Exception:
                # si el nombre de hoja no es un año, lo saltamos
                continue
            años[año] = df

    return años

def calc_percentile_day(jd, emeac, p):
    """Devuelve JD donde EMEAC cruza el percentil p (0–1)."""
    jd = np.asarray(jd, float)
    emeac = np.asarray(emeac, float)
    if emeac.max() < p:
        return np.nan
    return np.interp(p, emeac, jd)

# ===============================================================
# CARGA ANN (cacheado)
# ===============================================================
@st.cache_resource
def load_ann():
    return ANN()

ann = load_ann()

# ===============================================================
# SIDEBAR — ENTRADA DE ARCHIVOS
# ===============================================================
st.sidebar.header("Opciones de entrada")

uploaded_file = st.sidebar.file_uploader(
    "Subir archivo meteorológico (CSV: 1 año, XLSX: múltiples años)",
    type=["csv", "xlsx"],
    help="CSV: un año. XLSX: una hoja por año (ej. Bordenave_1977_2015_por_anio_con_JD.xlsx)."
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

# ===============================================================
# FLUJO PRINCIPAL
# ===============================================================
if uploaded_file is None:
    st.info("📄 Subí un archivo CSV o un XLSX con múltiples años para comenzar.")
    st.stop()

# Caso CSV: un solo año
if uploaded_file.name.lower().endswith(".csv"):
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception:
        st.error("No pude leer el archivo CSV. Verificá el formato.")
        st.stop()
    años = {"CSV": sanitize_meteo(df_raw)}

# Caso XLSX: múltiples hojas (una por año)
else:
    try:
        años = cargar_multiples_anios_xlsx(uploaded_file)
    except Exception as e:
        st.error(f"No se pudo leer el XLSX: {e}")
        st.stop()

    if len(años) == 0:
        st.error("No se encontraron hojas válidas con columnas JD,TMIN,TMAX,Prec.")
        st.stop()

st.success(f"📘 Archivo cargado. Años detectados: {list(años.keys())}")

# Selector de año
año_sel = st.sidebar.selectbox("Seleccionar año a analizar", sorted(años.keys()))
df_meteo = años[año_sel].copy()

# ===============================================================
# PREDICCIÓN ANN PARA EL AÑO SELECCIONADO
# ===============================================================
emerrel, emeac = ann.predict_emerrel(df_meteo)
df_meteo["EMERREL"] = emerrel
df_meteo["EMEAC"] = emeac
df_meteo["MA5"] = df_meteo["EMERREL"].rolling(5, min_periods=1).mean()

JD = df_meteo["JD"].to_numpy()
JD25 = calc_percentile_day(JD, emeac, 0.25)
JD50 = calc_percentile_day(JD, emeac, 0.50)
JD75 = calc_percentile_day(JD, emeac, 0.75)
JD95 = calc_percentile_day(JD, emeac, 0.95)

# Clasificación METEO → PATRÓN
res_patron = predecir_patron(df_meteo)
PATRON = res_patron["clasificacion"]
PROBAS = res_patron["probabilidades"]

# ===============================================================
# SALIDA PRINCIPAL
# ===============================================================
st.header(f"📊 Resultado de la clasificación — Año: {año_sel}")

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

# ===============================================================
# GRÁFICO EMERREL
# ===============================================================
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

# ===============================================================
# GRÁFICO EMEAC
# ===============================================================
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

# ===============================================================
# TABLA Y DESCARGA PARA EL AÑO SELECCIONADO
# ===============================================================
st.subheader("📥 Tabla de resultados (año seleccionado)")

tabla = df_meteo[["JD", "TMIN", "TMAX", "Prec", "EMERREL", "EMEAC", "MA5"]].copy()
st.dataframe(tabla, use_container_width=True)

csv = tabla.to_csv(index=False)
st.download_button(
    "Descargar resultados (CSV) — año seleccionado",
    data=csv,
    file_name=f"PREDWEEM_v8_EMERREL_EMEAC_{año_sel}.csv",
    mime="text/csv"
)

# ===============================================================
# PROCESAR TODOS LOS AÑOS (SI HAY MÚLTIPLES)
# ===============================================================
if len(años) > 1:
    st.header("📈 Procesar TODOS los años del archivo")

    if st.button("Ejecutar ANN + Clasificación para todos los años"):
        registros = []
        for año, dfmet in sorted(años.items()):
            dfmet2 = dfmet.copy()
            emerrel_all, emeac_all = ann.predict_emerrel(dfmet2)
            dfmet2["EMERREL"] = emerrel_all
            dfmet2["EMEAC"] = emeac_all

            JD25_all = calc_percentile_day(dfmet2["JD"], emeac_all, 0.25)
            JD50_all = calc_percentile_day(dfmet2["JD"], emeac_all, 0.50)
            JD75_all = calc_percentile_day(dfmet2["JD"], emeac_all, 0.75)
            JD95_all = calc_percentile_day(dfmet2["JD"], emeac_all, 0.95)

            res_all = predecir_patron(dfmet2)
            patron_all = res_all["clasificacion"]
            probas_all = res_all["probabilidades"]
            prob_max = max(probas_all.values()) if probas_all else np.nan

            registros.append({
                "Año": año,
                "Patrón": patron_all,
                "Prob_max": prob_max,
                "JD25": JD25_all,
                "JD50": JD50_all,
                "JD75": JD75_all,
                "JD95": JD95_all
            })

        tabla_all = pd.DataFrame(registros).sort_values("Año")
        st.dataframe(tabla_all, use_container_width=True)

        csv_all = tabla_all.to_csv(index=False)
        st.download_button(
            "Descargar tabla completa (CSV) — todos los años",
            data=csv_all,
            file_name="PREDWEEM_v8_todos_los_años.csv",
            mime="text/csv"
        )

st.success("✅ PREDWEEM v8 ejecutado correctamente.")
