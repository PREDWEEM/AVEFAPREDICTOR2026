# ===============================================================
# 🌾 PREDWEEM v8 — AVEFA Predictor 2026 (CORREGIDO Y ROBUSTO)
# Clasificación meteorológica (Early/Int/Late/Extended)
# + ANN emergencia diaria/acumulada SIN DISTORSIÓN
# Compatible con meteo_daily.csv
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

st.title("🌾 PREDWEEM v8 — AVEFA Predictor 2026")
st.subheader("Clasificación meteorológica + Emergencia ANN (sin distorsión)")

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
# 🔵 1. DETECTOR ROBUSTO DE COLUMNAS
# ===============================================================
def _detect_meteo_columns(df):

    posibles_jd = ["Julian_days", "julian_days", "julian", "jd", "doy", "dayofyear", "dia"]
    posibles_tmin = ["tmin", "temperatura_minima", "temp_min", "t_min", "minima"]
    posibles_tmax = ["tmax", "temperatura_maxima", "temp_max", "t_max", "maxima"]
    posibles_prec = [
        "prec", "lluvia", "ppt", "prcp", "rain",
        "precipitacion", "precipitacion_pluviometrica", "pp", "pmm"
    ]
    posibles_fecha = ["fecha", "date"]

    def match(lista):
        for target in lista:
            for col in df.columns:
                if target.lower() in col.lower():
                    return col
        return None

    # 🔴 PARCHE: si existe Julian_days → usarlo como JD
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
# 🔵 2. FEATURES PARA CLASIFICACIÓN METEO
# ===============================================================
def _build_features_from_df_meteo(df):

    df2 = df.copy()
    c_jd, c_tmin, c_tmax, c_prec, c_fecha = _detect_meteo_columns(df2)

    # Si no hay JD pero sí fecha → derivarlo
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
# 🔵 3. ENTRENAMIENTO INTERNO DEL CLASIFICADOR METEO (CORREGIDO)
# ===============================================================
@st.cache_resource
def load_clf():

    # CENTROIDES reales existentes
    cent = joblib.load(BASE / "predweem_model_centroides.pkl")
    C = cent["centroides"]  # DataFrame: index = nombre de patrones

    # HISTÓRICO METEOROLÓGICO
    xls = pd.ExcelFile(BASE / "Bordenave_1977_2015_por_anio_con_JD.xlsx")

    registros = []

    for sheet in xls.sheet_names:
        dfy = pd.read_excel(xls, sheet)

        c_jd, c_tmin, c_tmax, c_prec, c_fecha = _detect_meteo_columns(dfy)
        dfy = dfy.dropna(subset=[c_jd])

        dfy["Tmed"] = (dfy[c_tmin] + dfy[c_tmax]) / 2

        # ==========================
        # CENTROIDES → asignar patrón correcto
        # ==========================
        jd = dfy[c_jd].to_numpy()
        jd_sorted = np.sort(jd)

        # percentiles aproximados (no depende de EMERAC histórica)
        d25 = jd_sorted[int(0.25 * len(jd_sorted))]
        d50 = jd_sorted[int(0.50 * len(jd_sorted))]
        d75 = jd_sorted[int(0.75 * len(jd_sorted))]
        d95 = jd_sorted[int(0.95 * len(jd_sorted))]

        v = np.array([d25, d50, d75, d95])

        # Asignación real de patrón según el centroide más cercano
        dist = ((C.values - v)**2).sum(axis=1)**0.5
        patron = C.index[np.argmin(dist)]

        # ==========================
        # FEATURES METEO
        # ==========================
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

        feats["patron"] = patron
        registros.append(feats)

    # Dataset final para entrenar
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
# 🔵 4. PREDICCIÓN DE PATRÓN METEO
# ===============================================================
def predecir_patron(df_meteo):
    model = load_clf()
    Xnew = _build_features_from_df_meteo(df_meteo)
    proba = model.predict_proba(Xnew)[0]
    clases = model.classes_
    pred = clases[np.argmax(proba)]
    return pred, dict(zip(clases, proba))


# ===============================================================
# 🔵 5. ANN — EMERGENCIA (CORREGIDA)
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
            z2 = self.LLW @ a1 + self.bLW
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
# 🔵 6. RADAR JD25–95
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
    ax.fill(ang, vals_patron,  alpha=0



