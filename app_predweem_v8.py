# ===============================================================
# 🌾 PREDWEEM v8 — AVEFA Predictor 2026 (CORREGIDO Y ROBUSTO)
# Clasificación meteorológica + ANN emergencia simulada
# SIN DISTORSIÓN — Compatible con meteo_daily.csv
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM v8 — AVEFA 2026", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 PREDWEEM v8 — AVEFA 2026 (versión corregida)")
st.subheader("Clasificación meteorológica + Emergencia simulada por ANN")

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 🔧 FUNCIONES SEGURAS
# ===============================================================
def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None


# ===============================================================
# 🔵 1. DETECTOR ROBUSTO DE COLUMNAS METEOROLÓGICAS
# ===============================================================
def _detect_meteo_columns(df):

    posibles_jd = ["julian_days", "julian day", "julian", "jd", "doy", "dayofyear", "dia"]
    posibles_tmin = ["tmin", "temperatura_minima", "temp_min", "t_min", "minima"]
    posibles_tmax = ["tmax", "temperatura_maxima", "temp_max", "t_max", "maxima"]
    posibles_prec = [
        "prec", "lluvia", "ppt", "prcp", "rain",
        "precipitacion_pluviometrica", "precipitacion", "pp", "pmm"
    ]
    posibles_fecha = ["fecha", "date"]

    def match(col_list):
        for target in col_list:
            for col in df.columns:
                if target.lower() in col.lower():
                    return col
        return None

    # 🔴 PARCHE PRIORITARIO: si existe Julian_days → es JD
    if "Julian_days" in df.columns:
        c_jd = "Julian_days"
    else:
        c_jd = match(posibles_jd)

    c_tmin = match(posibles_tmin)
    c_tmax = match(posibles_tmax)
    c_prec = match(posibles_prec)
    c_fecha = match(posibles_fecha)

    return c_jd, c_tmin, c_tmax, c_prec, c_fecha


# ===============================================================
# 🔵 2. EXTRACCIÓN DE FEATURES PARA CLASIFICACIÓN METEO
# ===============================================================
def _build_features_from_df_meteo(df):
    df2 = df.copy()
    c_jd, c_tmin, c_tmax, c_prec, c_fecha = _detect_meteo_columns(df2)

    # Si falta JD pero hay Fecha → derivarlo
    if c_jd is None and c_fecha is not None:
        df2[c_fecha] = pd.to_datetime(df2[c_fecha], errors="coerce")
        df2["JD"] = df2[c_fecha].dt.dayofyear
        c_jd = "JD"

    df2[c_jd] = pd.to_numeric(df2[c_jd], errors="coerce")
    df2[c_tmin] = pd.to_numeric(df2[c_tmin], errors="coerce")
    df2[c_tmax] = pd.to_numeric(df2[c_tmax], errors="coerce")
    df2[c_prec] = pd.to_numeric(df2[c_prec], errors="coerce")

    df2 = df2.dropna(subset=[c_jd])

    df2["Tmed"] = (df2[c_tmin] + df2[c_tmax]) / 2

    feats = {
        "Tmin_mean": df2[c_tmin].mean(),
        "Tmax_mean": df2[c_tmax].mean(),
        "Tmed_mean": df2["Tmed"].mean(),
        "Prec_total": df2[c_prec].sum(),
        "Prec_days_10mm": (df2[c_prec] >= 10).sum()
    }

    sub = df2[df2[c_jd] <= 121]
    feats["Tmed_FM"] = sub["Tmed"].mean()
    feats["Prec_FM"] = sub[c_prec].sum()

    return pd.DataFrame([feats])


# ===============================================================
# 🔵 3. ENTRENAMIENTO INTERNO DEL CLASIFICADOR METEOROLÓGICO
# ===============================================================
@st.cache_resource
def load_clf():
    cent = joblib.load(BASE / "predweem_model_centroides.pkl")

    # Leer todas las hojas del archivo histórico
    xls = pd.ExcelFile(BASE / "Bordenave_1977_2015_por_anio_con_JD.xlsx")
    registros = []

    for sheet in xls.sheet_names:
        dfy = pd.read_excel(xls, sheet)

        c_jd, c_tmin, c_tmax, c_prec, c_fecha = _detect_meteo_columns(dfy)
        dfy = dfy.dropna(subset=[c_jd])

        dfy["Tmed"] = (dfy[c_tmin] + dfy[c_tmax]) / 2

        feats = {
            "Tmin_mean": dfy[c_tmin].mean(),
            "Tmax_mean": dfy[c_tmax].mean(),
            "Tmed_mean": dfy["Tmed"].mean(),
            "Prec_total": dfy[c_prec].sum(),
            "Prec_days_10mm": (dfy[c_prec] >= 10).sum(),
        }

        sub = dfy[dfy[c_jd] <= 121]
        feats["Tmed_FM"] = sub["Tmed"].mean()
        feats["Prec_FM"] = sub[c_prec].sum()

        # patrón asociado al centroide correspondiente
        patron = cent["centroides"].index[0]  
        feats["patron"] = patron
        registros.append(feats)

    feat_df = pd.DataFrame(registros)
    X = feat_df.drop(columns=["patron"])
    y = feat_df["patron"]

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=600,
            learning_rate=0.03,
            random_state=42
        ))
    ])
    clf.fit(X, y)
    return clf


# ===============================================================
# 🔵 4. PREDICCIÓN DEL PATRÓN METEO
# ===============================================================
def predecir_patron(df_meteo):
    model = load_clf()
    Xnew = _build_features_from_df_meteo(df_meteo)
    proba = model.predict_proba(Xnew)[0]
    clases = model.classes_
    pred = clases[np.argmax(proba)]
    return pred, dict(zip(clases, proba))


# ===============================================================
# 🔵 5. ANN — PREDICCIÓN DE EMERGENCIA
# ===============================================================
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2*(X - self.input_min)/(self.input_max - self.input_min)-1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer) + 1)/2
        emerac = np.cumsum(emer)
        emerrel = np.diff(emerac, prepend=0)
        return emerrel, emerac


@st.cache_resource
def load_ann():
    IW = np.load(BASE/"IW.npy")
    bIW = np.load(BASE/"bias_IW.npy")
    LW = np.load(BASE/"LW.npy")
    bLW = np.load(BASE/"bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)


def postprocess_emergence(emerrel_raw, smooth=True, window=3, clip=True):
    emer = np.array(emerrel_raw)
    if clip:
        emer = np.maximum(emer, 0)
    if smooth and window > 1:
        k = np.ones(window)/window
        emer = np.convolve(emer, k, mode="same")
    emerac = np.cumsum(emer)
    return emer, emerac


# ===============================================================
# 🔵 6. RADAR JD25-95
# ===============================================================
def radar(vals_year, vals_patron, patron_name, year_sel):
    labels = ["d25","d50","d75","d95"]
    vals_year = list(vals_year) + [vals_year[0]]
    vals_patron = list(vals_patron) + [vals_patron[0]]

    ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    ang = np.concatenate([ang, [ang[0]]])

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(ang, vals_year, lw=3, label=f"Año {year_sel}", color="blue")
    ax.fill(ang, vals_year, alpha=0.25, color="blue")

    ax.plot(ang, vals_patron, lw=2, label=f"Patrón {patron_name}", color="green")
    ax.fill(ang, vals_patron, alpha=0.15, color="green")

    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Radar JD25-95", fontsize=14)
    ax.legend(loc="best")
    return fig


# ===============================================================
# 🔵 SIDEBAR ANN
# ===============================================================
with st.sidebar:
    st.header("Ajustes ANN")
    smooth = st.checkbox("Suavizar EMERREL", True)
    win = st.slider("Ventana", 1, 9, 3)
    clip = st.checkbox("Recorte negativos", True)


# ===============================================================
# 🔵 SUBIR ARCHIVO
# ===============================================================
uploaded = st.file_uploader("📤 Subir archivo meteorológico", type=["csv","xlsx"])

modelo_ann = safe(load_ann, "No se pudo cargar ANN")

if uploaded is not None:

    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("📄 Archivo cargado")
    st.dataframe(df)

    # Detectar años
    col_fecha = next((c for c in df.columns if "fecha" in c.lower()), None)
    col_anio = next((c for c in df.columns if c.lower() in ["año","ano","year"]), None)

    if col_fecha:
        df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
        years = sorted(df[col_fecha].dt.year.dropna().unique())
    elif col_anio:
        df[col_anio] = pd.to_numeric(df[col_anio], errors="coerce")
        years = sorted(df[col_anio].dropna().unique())
    else:
        years = [2025]

    modo = st.radio("¿Qué analizar?", ["Todos los años","Un año"])

    # ------------------------------------------------------------
    # MULTI-AÑO
    # ------------------------------------------------------------
    if modo == "Todos los años":
        resultados = []
        for y in years:
            if col_fecha:
                dfy = df[df[col_fecha].dt.year == y]
            else:
                dfy = df[df[col_anio] == y]

            patron, probs = predecir_patron(dfy)
            fila = {"Año": y, "Patrón": patron}
            for k,v in probs.items():
                fila[f"P_{k}"] = float(v)
            resultados.append(fila)

        tabla = pd.DataFrame(resultados)
        st.subheader("📊 Clasificación meteorológica por año")
        st.dataframe(tabla)


    # ------------------------------------------------------------
    # UN AÑO → METEO + ANN
    # ------------------------------------------------------------
    else:
        year_sel = st.selectbox("Seleccionar año", years)

        if col_fecha:
            dfy = df[df[col_fecha].dt.year == year_sel].copy()
            dfy = dfy.sort_values(col_fecha)
        else:
            dfy = df[df[col_anio] == year_sel].copy()

        st.subheader(f"📅 Año {year_sel}")
        st.dataframe(dfy)

        # === PATRÓN METEO ===
        patron_pred, probs = predecir_patron(dfy)
        st.markdown(f"## 🌱 Patrón meteorológico: **{patron_pred}**")
        st.json(probs)

        # =======================================================
        # ANN — Emergencia
        # =======================================================
        st.subheader("🌾 Emergencia ANN")

        c_jd, c_tmin, c_tmax, c_prec, c_fecha = _detect_meteo_columns(dfy)
        df_ann = dfy.copy()

        # Si falta JD pero hay Fecha
        if c_jd is None and c_fecha is not None:
            df_ann[c_fecha] = pd.to_datetime(df_ann[c_fecha], errors="coerce")
            df_ann["JD"] = df_ann[c_fecha].dt.dayofyear
            c_jd = "JD"

        df_ann[c_jd] = pd.to_numeric(df_ann[c_jd], errors="coerce")
        df_ann[c_tmin] = pd.to_numeric(df_ann[c_tmin], errors="coerce")
        df_ann[c_tmax] = pd.to_numeric(df_ann[c_tmax], errors="coerce")
        df_ann[c_prec] = pd.to_numeric(df_ann[c_prec], errors="coerce")

        df_ann = df_ann.dropna(subset=[c_jd])
        df_ann = df_ann.sort_values(c_jd)

        dias = df_ann[c_jd].to_numpy()
        X_ann = df_ann[[c_jd, c_tmax, c_tmin, c_prec]].to_numpy(float)

        emerrel_raw, emerac_raw = modelo_ann.predict(X_ann)
        emerrel, emerac = postprocess_emergence(emerrel_raw, smooth, win, clip)

        df_ann["EMERREL"] = emerrel
        df_ann["EMERAC"] = emerac

        # --- GRAFICOS ---
        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(5,4))
            ax1.plot(dias, emerrel_raw, label="Cruda", color="red")
            ax1.plot(dias, emerrel, label="Procesada", color="blue")
            ax1.set_title("EMERREL ANN")
            ax1.legend()
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.plot(dias, emerac_raw/emerac_raw[-1], label="Cruda", color="orange")
            ax2.plot(dias, emerac/emerac[-1], label="Procesada", color="green")
            ax2.set_title("EMERAC ANN")
            ax2.legend()
            st.pyplot(fig2)

        # =======================================================
        # Percentiles ANN
        # =======================================================
        st.subheader("📌 Percentiles ANN (d25–d95)")

        if emerac.max() > 0:
            y = emerac / emerac.max()
            d25 = np.interp(0.25, y, dias)
            d50 = np.interp(0.50, y, dias)
            d75 = np.interp(0.75, y, dias)
            d95 = np.interp(0.95, y, dias)

            st.write({
                "d25": round(d25,1),
                "d50": round(d50,1),
                "d75": round(d75,1),
                "d95": round(d95,1)
            })

            # Radar vs patrón
            cent = joblib.load(BASE/"predweem_model_centroides.pkl")
            C = cent["centroides"]

            if patron_pred in C.index:
                vals_patron = list(C.loc[patron_pred][["JD25","JD50","JD75","JD95"]])
                fig_rad = radar([d25,d50,d75,d95], vals_patron, patron_pred, year_sel)
                st.pyplot(fig_rad)
            else:
                st.warning("El patrón predicho no existe en los centroides.")
        else:
            st.error("No se pudieron calcular percentiles.")

        # DESCARGA ANN
        st.download_button(
            "⬇️ Descargar EMERGENCIA ANN",
            df_ann.to_csv(index=False).encode("utf-8"),
            f"emergencia_ANN_{year_sel}.csv"
        )


