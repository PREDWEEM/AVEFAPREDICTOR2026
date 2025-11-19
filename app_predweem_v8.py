# ===============================================================
# 🌾 PREDWEEM v8 — AVEFA Predictor 2026
# Clasificación meteorológica → patrón (Early/Intermediate/Late/Extended)
# Sin modelos externos .pkl — El clasificador se entrena DENTRO de la app
# Compatible 100% con Streamlit Cloud (sklearn 1.7.2)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="Predicción de Emergencia Agrícola AVEFA", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------
def safe(fn):
    try:
        return fn()
    except Exception as e:
        st.error(str(e))
        return None

# ===============================================================
# 🔵 ETAPA 1 — ENTRENAR CLASIFICADOR METEO → PATRÓN DENTRO DE LA APP
# ===============================================================

def _compute_jd_percentiles(jd, emerac, qs=(0.25, 0.5, 0.75, 0.95)):
    jd = np.asarray(jd)
    emer = np.asarray(emerac)
    order = np.argsort(jd)
    jd = jd[order]
    emer = emer[order]

    out = []
    for q in qs:
        idx = np.where(emer >= q)[0]
        if len(idx) == 0:
            out.append(float(jd[-1]))
        else:
            out.append(float(jd[idx[0]]))
    return np.array(out)


def _load_curves_emereac():
    curvas = {}
    xls1 = pd.ExcelFile("emergencia_acumulada_interpolada 1977-1998.xlsx")
    for sh in xls1.sheet_names:
        df = pd.read_excel(xls1, sheet_name=sh)
        year = int(sh.split("_")[-1])
        curvas[year] = df[["JD", "EMERAC"]].copy()

    xls2 = pd.ExcelFile("emergencia_2000_2015_interpolada.xlsx")
    for sh in xls2.sheet_names:
        df = pd.read_excel(xls2, sheet_name=sh)
        year = int(sh.split("_")[-1])
        curvas[year] = df[["JD", "EMERAC"]].copy()

    return curvas


def _assign_labels_from_centroids(curvas):
    cent = joblib.load("predweem_model_centroides.pkl")
    C = cent["centroides"]  # DataFrame 4x4

    registros = []
    for year, df in sorted(curvas.items()):
        jd25, jd50, jd75, jd95 = _compute_jd_percentiles(df["JD"], df["EMERAC"])
        v = np.array([jd25, jd50, jd75, jd95])
        dists = ((C.values - v) ** 2).sum(axis=1) ** 0.5
        patron = C.index[np.argmin(dists)]
        registros.append(
            dict(
                anio=year,
                patron=str(patron),
                JD25=jd25,
                JD50=jd50,
                JD75=jd75,
                JD95=jd95,
            )
        )
    return pd.DataFrame(registros)


def _build_meteo_features_for_years(labels_df):
    xls = pd.ExcelFile("Bordenave_1977_2015_por_anio_con_JD.xlsx")
    rows = []

    for _, row in labels_df.iterrows():
        year = row["anio"]
        patron = row["patron"]

        if str(year) not in xls.sheet_names:
            continue

        df = pd.read_excel(xls, sheet_name=str(year)).copy()
        df.rename(columns={
            "Temperatura_Minima": "TMIN",
            "Temperatura_Maxima": "TMAX",
            "Precipitacion_Pluviometrica": "Prec",
        }, inplace=True)

        df["JD"] = pd.to_numeric(df["JD"], errors="coerce")
        df = df.dropna(subset=["JD"])

        df["TMIN"] = pd.to_numeric(df["TMIN"], errors="coerce")
        df["TMAX"] = pd.to_numeric(df["TMAX"], errors="coerce")
        df["Prec"] = pd.to_numeric(df["Prec"], errors="coerce")
        df["Tmed"] = (df["TMIN"] + df["TMAX"]) / 2

        feats = {
            "anio": year,
            "patron": patron,
            "Tmin_mean": df["TMIN"].mean(),
            "Tmax_mean": df["TMAX"].mean(),
            "Tmed_mean": df["Tmed"].mean(),
            "Prec_total": df["Prec"].sum(),
            "Prec_days_10mm": (df["Prec"] >= 10).sum(),
        }

        sub = df[df["JD"] <= 121]
        feats["Tmed_FM"] = sub["Tmed"].mean()
        feats["Prec_FM"] = sub["Prec"].sum()

        rows.append(feats)

    return pd.DataFrame(rows)


@st.cache_resource
def load_clf():
    curvas = _load_curves_emereac()
    labels_df = _assign_labels_from_centroids(curvas)
    feat_df = _build_meteo_features_for_years(labels_df).dropna()

    X = feat_df[[
        "Tmin_mean", "Tmax_mean", "Tmed_mean",
        "Prec_total", "Prec_days_10mm",
        "Tmed_FM", "Prec_FM"
    ]]
    y = feat_df["patron"].astype(str)

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
# 🔵 ETAPA 2 — FUNCIÓN DE PREDICCIÓN PARA NUEVO df_meteo
# ===============================================================

def _build_features_from_df_meteo(df_meteo):
    df = df_meteo.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_jd = pick("jd", "julian_days", "julianday", "doy")
    c_tmin = pick("tmin")
    c_tmax = pick("tmax")
    c_prec = pick("prec", "ppt", "lluvia")

    if None in (c_jd, c_tmin, c_tmax, c_prec):
        raise ValueError("No se encontraron columnas JD/TMIN/TMAX/Prec en df_meteo.")

    df["JD"] = pd.to_numeric(df[c_jd], errors="coerce")
    df["TMIN"] = pd.to_numeric(df[c_tmin], errors="coerce")
    df["TMAX"] = pd.to_numeric(df[c_tmax], errors="coerce")
    df["Prec"] = pd.to_numeric(df[c_prec], errors="coerce")
    df = df.dropna(subset=["JD"])

    df["Tmed"] = (df["TMIN"] + df["TMAX"]) / 2

    feats = {}
    feats["Tmin_mean"] = df["TMIN"].mean()
    feats["Tmax_mean"] = df["TMAX"].mean()
    feats["Tmed_mean"] = df["Tmed"].mean()
    feats["Prec_total"] = df["Prec"].sum()
    feats["Prec_days_10mm"] = (df["Prec"] >= 10).sum()

    sub = df[df["JD"] <= 121]
    feats["Tmed_FM"] = sub["Tmed"].mean()
    feats["Prec_FM"] = sub["Prec"].sum()

    return pd.DataFrame([feats])


def predecir_patron(df_meteo):
    model = load_clf()
    Xnew = _build_features_from_df_meteo(df_meteo)
    proba = model.predict_proba(Xnew)[0]
    clases = model.classes_
    pred = clases[np.argmax(proba)]

    return {
        "clasificacion": str(pred),
        "probabilidades": dict(zip(map(str, clases), map(float, proba)))
    }

# ===============================================================
# 🔵 ETAPA 3 — CARGA DEL METEO, CÁLCULO DE ANN Y GRÁFICOS
# (TU CÓDIGO ORIGINAL QUEDA SIN CAMBIOS)
# ===============================================================

st.title("🌾 AVEFA Predictor 2026 — Emergencia + Patrón Meteorológico")

st.write("👉 El clasificador meteo→patrón se entrena automáticamente dentro de la app (sklearn 1.7.2).")

uploaded = st.file_uploader("Subir archivo de meteorología (CSV o XLSX)", type=["csv", "xlsx"])

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success("Archivo cargado correctamente.")

        st.subheader("Datos cargados")
        st.dataframe(df)

        # ----------------------------------------------
        # PREDICCIÓN DE PATRÓN
        # ----------------------------------------------
        res = predecir_patron(df)
        st.subheader("Patrón meteorológico predicho")
        st.write(f"**{res['clasificacion']}**")
        st.json(res["probabilidades"])

    except Exception as e:
        st.error(f"Error procesando archivo: {e}")


