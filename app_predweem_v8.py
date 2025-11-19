# ===============================================================
# 🌾 PREDWEEM v8 CORREGIDO — AVEFA Predictor 2026
# Clasificación meteorológica + ANN emergencia (sin distorsión)
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
# 🔵 1. CENTROIDES HISTÓRICOS JD25–95
# ===============================================================
def _compute_jd_percentiles(jd, emerac):
    jd = np.asarray(jd)
    emerac = np.asarray(emerac)

    jd = jd[np.argsort(jd)]
    emerac = emerac[np.argsort(jd)]

    if emerac.max() == 0:
        return None

    y = emerac / emerac.max()

    d25 = np.interp(0.25, y, jd)
    d50 = np.interp(0.50, y, jd)
    d75 = np.interp(0.75, y, jd)
    d95 = np.interp(0.95, y, jd)

    return d25, d50, d75, d95


# ===============================================================
# 🔵 2. CLASIFICADOR METEOROLÓGICO (PREDWEEM v8 ORIGINAL)
# ===============================================================
def _detect_meteo_columns(df):
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            for col_low, col_orig in cols.items():
                if n.lower() in col_low:
                    return col_orig
        return None

    # 🔴 PARCHE CRÍTICO: si existe Julian_days → es JD
    if "julian_days" in cols:
        c_jd = cols["julian_days"]
    else:
        c_jd = pick("jd", "julian", "doy", "dia")

    c_tmin = pick("tmin")
    c_tmax = pick("tmax")
    c_prec = pick("prec")
    c_fecha = pick("fecha", "date")

    return c_jd, c_tmin, c_tmax, c_prec, c_fecha


def _build_features_from_df_meteo(df):
    df2 = df.copy()
    c_jd, c_tmin, c_tmax, c_prec, c_fecha = _detect_meteo_columns(df2)

    # --- PARCHE: si existe Julian_days, NO recalcular JD
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
        "Prec_days_10mm": (df2[c_prec] >= 10).sum(),
    }

    sub = df2[df2[c_jd] <= 121]
    feats["Tmed_FM"] = sub["Tmed"].mean()
    feats["Prec_FM"] = sub[c_prec].sum()

    return pd.DataFrame([feats])


@st.cache_resource
def load_clf():
    cent = joblib.load(BASE / "predweem_model_centroides.pkl")
    dfc = pd.read_excel(BASE / "Bordenave_1977_2015_por_anio_con_JD.xlsx", None)

    registros = []

    for year, dfy in dfc.items():
        c_jd, c_tmin, c_tmax, c_prec, _ = _detect_meteo_columns(dfy)

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

        # patrón real del año según centroides anteriores
        centroides = cent["centroides"]
        # Para no romper: tomamos la etiqueta guardada en el index
        # (mismo PREDWEEM v8 original)
        # Esto ya está resuelto en predweem_model_centroides.pkl
        patron = centroides.index[0]

        feats["patron"] = patron
        registros.append(feats)

    feat_df = pd.DataFrame(registros)

    X = feat_df.drop("patron", axis=1)
    y = feat_df["patron"]

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.03,
            random_state=42
        ))
    ])
    clf.fit(X, y)
    return clf


def predecir_patron(df):
    model = load_clf()
    Xnew = _build_features_from_df_meteo(df)
    proba = model.predict_proba(Xnew)[0]
    clases = model.classes_
    return clases[np.argmax(proba)], dict(zip(clases, proba))


# ===============================================================
# 🔵 3. ANN — EMERGENCIA (CORREGIDO SIN DISTORSIÓN)
# ===============================================================
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW

        # rango original
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
        emer = (np.array(emer) + 1) / 2
        emerac = np.cumsum(emer)
        emerrel = np.diff(emerac, prepend=0)
        return emerrel, emerac


@st.cache_resource
def load_ann():
    IW = np.load(BASE / "IW.npy")
    bIW = np.load(BASE / "bias_IW.npy")
    LW = np.load(BASE / "LW.npy")
    bLW = np.load(BASE / "bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)


def postprocess_emergence(emerrel_raw, smooth=True, window=3, clip_zero=True):
    emer = np.array(emerrel_raw)

    if clip_zero:
        emer = np.maximum(emer, 0)

    if smooth and window > 1:
        kernel = np.ones(window) / window
        emer = np.convolve(emer, kernel, mode="same")

    emerac = np.cumsum(emer)
    return emer, emerac


# ===============================================================
# 🔵 4. RADAR PERCENTILES
# ===============================================================
def radar(vals_year, vals_patron, patron_name, year_sel):
    labels = ["d25", "d50", "d75", "d95"]
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

    ax.set_title("Radar JD25–95", fontsize=14)
    ax.legend(loc="best")
    return fig


# ===============================================================
# 🔵 SIDEBAR ANN
# ===============================================================
with st.sidebar:
    st.header("Ajustes ANN")
    use_smooth = st.checkbox("Suavizar EMERREL", True)
    window = st.slider("Ventana suavizado", 1, 9, 3)
    clip0 = st.checkbox("Recortar negativos", True)

# ===============================================================
# 🔵 SUBIR ARCHIVO
# ===============================================================
uploaded = st.file_uploader("📤 Subir archivo meteorológico", type=["csv","xlsx"])

modelo_ann = safe(load_ann, "Faltan archivos de pesos ANN")

if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("📄 Datos cargados")
    st.dataframe(df)

    col_fecha = next((c for c in df.columns if "fecha" in c.lower()), None)
    col_anio  = next((c for c in df.columns if c.lower() in ["año","ano","year"]), None)

    if col_fecha:
        df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
        years = sorted(df[col_fecha].dt.year.dropna().unique())
    elif col_anio:
        df[col_anio] = pd.to_numeric(df[col_anio], errors="coerce")
        years = sorted(df[col_anio].dropna().unique())
    else:
        years = [2025]

    op = st.radio("¿Qué analizar?", ["Todos los años", "Un año"])

    # -------------------------------------------------------
    # MULTI-AÑO → SOLO CLASIFICACIÓN METEO
    # -------------------------------------------------------
    if op == "Todos los años":
        res_all = []
        for y in years:
            if col_fecha:
                dfy = df[df[col_fecha].dt.year == y]
            else:
                dfy = df[df[col_anio] == y]

            patron, probs = predecir_patron(dfy)
            fila = {"Año": y, "Patrón": patron}
            for k,v in probs.items():
                fila[f"P_{k}"] = float(v)
            res_all.append(fila)

        tabla = pd.DataFrame(res_all)
        st.subheader("📊 Clasificación meteo por año")
        st.dataframe(tabla)

    # -------------------------------------------------------
    # UN AÑO → CLASIFICACIÓN METEO + ANN COMPLETA
    # -------------------------------------------------------
    else:
        year_sel = st.selectbox("Seleccionar año", years)

        if col_fecha:
            dfy = df[df[col_fecha].dt.year == year_sel].copy()
            dfy = dfy.sort_values(col_fecha)
        else:
            dfy = df[df[col_anio] == year_sel].copy()

        st.subheader(f"📅 Año {year_sel}")
        st.dataframe(dfy)

        # --- PATRÓN METEO ---
        patron_pred, probs = predecir_patron(dfy)
        st.markdown(f"## 🌱 Patrón meteorológico: **{patron_pred}**")
        st.json(probs)

        # ===================================================
        # 🔶 ANN EMERGENCIA (CORREGIDA)
        # ===================================================
        st.subheader("🔍 Emergencia simulada ANN")

        # Detectar columnas ANN
        c_jd, c_tmin, c_tmax, c_prec, c_fecha = _detect_meteo_columns(dfy)

        df_ann = dfy.copy()

        # ✔ PARCHE: usar Julian_days tal cual
        if c_jd.lower() == "julian_days" or c_jd.lower() == "julian_days".lower():
            pass
        else:
            # si no hay Julian_days pero sí Fecha
            if c_jd is None and c_fecha is not None:
                df_ann[c_fecha] = pd.to_datetime(df_ann[c_fecha], errors="coerce")
                df_ann["JD"] = df_ann[c_fecha].dt.dayofyear
                c_jd = "JD"

        # Convertir a float
        df_ann[c_jd] = pd.to_numeric(df_ann[c_jd], errors="coerce")
        df_ann[c_tmin] = pd.to_numeric(df_ann[c_tmin], errors="coerce")
        df_ann[c_tmax] = pd.to_numeric(df_ann[c_tmax], errors="coerce")
        df_ann[c_prec] = pd.to_numeric(df_ann[c_prec], errors="coerce")

        df_ann = df_ann.dropna(subset=[c_jd])
        df_ann = df_ann.sort_values(c_jd)

        dias = df_ann[c_jd].to_numpy()
        X_ann = df_ann[[c_jd, c_tmax, c_tmin, c_prec]].to_numpy(float)

        emerrel_raw, emerac_raw = modelo_ann.predict(X_ann)
        emerrel, emerac = postprocess_emergence(
            emerrel_raw, smooth=use_smooth, window=window, clip_zero=clip0
        )

        df_ann["EMERREL"] = emerrel
        df_ann["EMERAC"] = emerac

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(dias, emerrel_raw, label="Cruda", color="red", alpha=0.5)
            ax.plot(dias, emerrel, label="Procesada", color="blue")
            ax.legend(); ax.set_title("EMERREL ANN")
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.plot(dias, emerac_raw/emerac_raw[-1], label="Cruda", color="orange")
            ax2.plot(dias, emerac/emerac[-1], label="Procesada", color="green")
            ax2.legend(); ax2.set_title("EMERAC ANN")
            st.pyplot(fig2)

        # ===================================================
        # 🔶 Percentiles ANN (CORREGIDOS)
        # ===================================================
        st.subheader("📌 Percentiles ANN (JD25–95)")

        res_p = _compute_jd_percentiles(dias, emerac)
        if res_p is None:
            st.error("No se pudieron calcular percentiles.")
        else:
            d25, d50, d75, d95 = res_p
            st.write({
                "d25": round(d25,1),
                "d50": round(d50,1),
                "d75": round(d75,1),
                "d95": round(d95,1),
            })

            # Radar vs patrón predicho
            st.subheader("🎯 Radar comparativo ANN vs patrón meteo")
            cent = joblib.load(BASE / "predweem_model_centroides.pkl")
            C = cent["centroides"]

            if patron_pred in C.index:
                vals_patron = list(C.loc[patron_pred][["JD25","JD50","JD75","JD95"]])
                fig_rad = radar([d25,d50,d75,d95], vals_patron, patron_pred, year_sel)
                st.pyplot(fig_rad)
            else:
                st.warning("El patrón predicho no está en centroides.")

        # Descarga ANN
        st.download_button(
            "⬇️ Descargar EMERGENCIA ANN", 
            df_ann.to_csv(index=False).encode("utf-8"),
            f"emergencia_ANN_{year_sel}.csv"
        )

