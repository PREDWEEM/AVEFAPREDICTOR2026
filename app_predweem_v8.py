
# ===============================================================
# 🌾 PREDWEEM v8.5 — AVEFA Predictor 2026 
# ANN + Centroides + Reglas agronómicas avanzadas
# + Módulo de diagnóstico visual avanzado
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ---------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM v8.5 — AVEFA 2026", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 PREDWEEM v8.5 — AVEFA Predictor 2026")
st.subheader("ANN + Centroides + Reglas fisiológicas avanzadas")

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()


# ===============================================================
# FUNCIONES AUXILIARES
# ===============================================================

def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

def rmse_curvas(y_pred, y_obs):
    y_pred = np.asarray(y_pred, float)
    y_obs  = np.asarray(y_obs, float)
    return float(np.sqrt(np.mean((y_pred - y_obs) ** 2)))


# ===============================================================
# COMPUTO DE PERCENTILES
# ===============================================================

def _compute_jd_percentiles(jd, emerac, qs=(0.25, 0.5, 0.75, 0.95)):
    jd = np.asarray(jd, float)
    emer = np.asarray(emerac, float)

    order = np.argsort(jd)
    jd = jd[order]
    emer = emer[order]

    if emer.max() <= 0:
        return None

    y = emer / emer.max()
    return np.array([np.interp(q, y, jd) for q in qs], float)


# ===============================================================
# CARGA HISTÓRICA + CENTROIDES
# ===============================================================

def _load_curves_emereac():
    curvas = {}
    try:
        xls1 = pd.ExcelFile(BASE / "emergencia_acumulada_interpolada 1977-1998.xlsx")
        for sh in xls1.sheet_names:
            df = pd.read_excel(xls1, sh)
            if "JD" not in df.columns or "EMERAC" not in df.columns:
                continue
            year = int(str(sh).split("_")[-1])
            curvas[year] = df[["JD", "EMERAC"]].dropna()
    except Exception:
        pass

    try:
        xls2 = pd.ExcelFile(BASE / "emergencia_2000_2015_interpolada.xlsx")
        for sh in xls2.sheet_names:
            df = pd.read_excel(xls2, sh)
            if "JD" not in df.columns or "EMERAC" not in df.columns:
                continue
            year = int(str(sh).split("_")[-1])
            curvas[year] = df[["JD", "EMERAC"]].dropna()
    except Exception:
        pass

    return curvas

def _assign_labels_from_centroids(curvas, C):
    regs = []
    for year, df in curvas.items():
        vals = _compute_jd_percentiles(df["JD"], df["EMERAC"])
        if vals is None:
            continue
        d25, d50, d75, d95 = vals
        v = np.array(vals)
        dists = np.linalg.norm(C.values - v, axis=1)
        patron = C.index[np.argmin(dists)]
        regs.append({
            "anio": int(year),
            "patron": str(patron),
            "JD25": d25,
            "JD50": d50,
            "JD75": d75,
            "JD95": d95
        })
    return pd.DataFrame(regs)

@st.cache_resource
def load_centroides_y_historia():
    cent = joblib.load(BASE / "predweem_model_centroides.pkl")
    C = cent["centroides"]
    curvas = _load_curves_emereac()
    labels = _assign_labels_from_centroids(curvas, C)

    rep_year = {}
    for pat in C.index:
        sub = labels[labels["patron"] == str(pat)]
        if sub.empty:
            continue
        vc = C.loc[pat][["JD25", "JD50", "JD75", "JD95"]].values.astype(float)
        M = sub[["JD25", "JD50", "JD75", "JD95"]].values.astype(float)
        best = np.argmin(np.linalg.norm(M - vc, axis=1))
        rep_year[str(pat)] = int(sub.iloc[best]["anio"])
    return C, labels, rep_year, curvas


# ===============================================================
# ANN
# ===============================================================

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
        # Rango de normalización [JD, TMAX, TMIN, Prec]
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, X):
        Xn = self.normalize(X)
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
    IW  = np.load(BASE / "IW.npy")
    bIW = np.load(BASE / "bias_IW.npy")
    LW  = np.load(BASE / "LW.npy")
    bLW = np.load(BASE / "bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)

def postprocess_emergence(raw, smooth=True, window=3, clip_zero=True):
    emer = np.maximum(raw, 0.0) if clip_zero else raw
    if smooth and window > 1:
        k = np.ones(int(window)) / int(window)
        emer = np.convolve(emer, k, mode="same")
    emerac = np.cumsum(emer)
    return emer, emerac


# ===============================================================
# REGLAS AGRONÓMICAS
# ===============================================================

def aplicar_reglas_agronomicas(JD_ini, JD25, JD50, JD75, JD95, patron_inicial=None):
    """
    Devuelve el patrón 'override' si alguna regla lo define claramente.
    Si no hay override, devuelve None y se usa la clasificación por centroides.
    """

    banda = JD75 - JD25

    # EXTENDED (prioridad absoluta)
    if (50 <= JD_ini <= 80) and (JD50 > 150) and (banda > 120):
        return "Extended"

    # EARLY
    if (JD_ini < 70) and (JD50 < 140) and (banda < 60):
        return "Early"

    # INTERMEDIATE
    if (70 <= JD_ini <= 110) and (130 <= JD50 <= 160) and (70 <= banda <= 90):
        return "Intermediate"

    # LATE
    if (JD_ini > 110) and (JD50 > 160):
        return "Late"

    return None


def clasificar_patron_desde_ann(dias, emerac, C):
    """
    ANN -> EMERAC -> percentiles -> reglas agronómicas + centroides.
    """

    vals = _compute_jd_percentiles(dias, emerac)
    if vals is None:
        return None, None, None, None

    d25, d50, d75, d95 = vals
    v = np.array([d25, d50, d75, d95], float)

    # Distancias a centroides
    dists = np.linalg.norm(C.values - v, axis=1)
    w = 1.0 / (dists + 1e-6)
    p = w / w.sum()
    prob_base = {str(C.index[i]): float(p[i]) for i in range(len(C.index))}

    emerac = np.asarray(emerac, float)
    dias = np.asarray(dias, float)

    idx = np.where(emerac > 0.01)[0]
    JD_ini = dias[idx[0]] if len(idx) > 0 else np.inf

    # Reglas agronómicas
    override = aplicar_reglas_agronomicas(JD_ini, d25, d50, d75, d95, None)

    if override is not None and override in C.index:
        patron_final = override
        d_mod = dists.copy()
        idxp = list(C.index).index(patron_final)
        d_mod[idxp] = -1.0  # forzamos que sea el más cercano

        w = 1.0 / (d_mod - d_mod.min() + 1e-6)
        p = w / w.sum()
        prob_mod = {str(C.index[i]): float(p[i]) for i in range(len(C.index))}
        return patron_final, vals, d_mod, prob_mod

    # Sin override → centroide más cercano
    idx_min = int(np.argmin(dists))
    patron = str(C.index[idx_min])
    return patron, vals, dists, prob_base


# ===============================================================
# NORMALIZAR ARCHIVO METEO
# ===============================================================

def normalizar_meteo(df):
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            for k, v in cols.items():
                if n in k:
                    return v
        return None

    c_jd   = pick("jd", "julian", "dia")
    c_tmax = pick("tmax")
    c_tmin = pick("tmin")
    c_prec = pick("prec", "lluv")
    c_fecha= pick("fecha")

    if None in (c_jd, c_tmax, c_tmin, c_prec):
        raise ValueError("No se identifican JD / TMAX / TMIN / Prec en el archivo meteorológico.")

    df["JD"]   = pd.to_numeric(df[c_jd],   errors="coerce")
    df["TMAX"] = pd.to_numeric(df[c_tmax], errors="coerce")
    df["TMIN"] = pd.to_numeric(df[c_tmin], errors="coerce")
    df["Prec"] = pd.to_numeric(df[c_prec], errors="coerce")
    if c_fecha is not None:
        df["Fecha"] = pd.to_datetime(df[c_fecha], errors="coerce")
    else:
        df["Fecha"] = pd.NaT

    df = df.dropna(subset=["JD"])
    return df.sort_values("JD")


# ===============================================================
# MÓDULO: ESTADÍSTICAS DE PATRONES PARA DIAGNÓSTICO VISUAL
# ===============================================================

def build_patron_stats(C, labels_df, curvas_hist, jd_min=1, jd_max=365):
    """
    Construye, para cada patrón, curvas:
      - mediana
      - p25
      - p75
    sobre una grilla común de JD.
    """
    grid = np.arange(jd_min, jd_max + 1)
    stats = {}
    for patron in C.index:
        sub = labels_df[labels_df["patron"] == str(patron)]
        series = []
        for _, row in sub.iterrows():
            year = int(row["anio"])
            if year not in curvas_hist:
                continue
            df = curvas_hist[year]
            jd = df["JD"].to_numpy(float)
            em = df["EMERAC"].to_numpy(float)
            if np.nanmax(em) <= 0:
                continue
            em_norm = em / np.nanmax(em)
            series.append(np.interp(grid, jd, em_norm))
        if len(series) == 0:
            continue
        A = np.vstack(series)
        stats[str(patron)] = {
            "grid": grid,
            "median": np.nanmedian(A, axis=0),
            "p25":    np.nanpercentile(A, 25, axis=0),
            "p75":    np.nanpercentile(A, 75, axis=0),
        }
    return stats


# ===============================================================
# SIDEBAR
# ===============================================================

with st.sidebar:
    st.header("Ajustes ANN")
    smooth   = st.checkbox("Suavizar EMERREL", True)
    window   = st.slider("Ventana de suavizado (días)", 1, 9, 3)
    clip     = st.checkbox("Recortar valores negativos a 0", True)


# ===============================================================
# CARGA METEO
# ===============================================================

st.subheader("📤 Cargar archivo meteorológico")
uploaded = st.file_uploader("Archivo CSV o XLSX con JD, TMAX, TMIN, Prec", type=["csv", "xlsx"])

modelo_ann = safe(load_ann, "Error cargando pesos de la ANN")
if uploaded is None:
    st.info("Suba un archivo meteorológico para iniciar el análisis.")
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
st.success("Archivo meteorológico cargado.")
st.dataframe(df_raw, use_container_width=True)

try:
    df_meteo = normalizar_meteo(df_raw)
except Exception as e:
    st.error(f"Error normalizando el archivo meteorológico: {e}")
    st.stop()

if modelo_ann is None:
    st.stop()


# ===============================================================
# ANN → EMERREL / EMERAC
# ===============================================================

st.subheader("🔍 Emergencia simulada por ANN (EMERREL / EMERAC)")

df_ann = df_meteo.copy()
dias   = df_ann["JD"].to_numpy(float)
X      = df_ann[["JD", "TMAX", "TMIN", "Prec"]].to_numpy(float)

emerrel_raw, emerac_raw = modelo_ann.predict(X)
emerrel, emerac = postprocess_emergence(emerrel_raw, smooth=smooth, window=window, clip_zero=clip)

df_ann["EMERREL"] = emerrel
df_ann["EMERAC"]  = emerac

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(dias, emerrel_raw, label="EMERREL cruda", color="red", alpha=0.5)
    ax.plot(dias, emerrel,     label="EMERREL procesada", color="blue", linewidth=2)
    ax.set_xlabel("Día Juliano")
    ax.set_ylabel("EMERREL (fracción diaria)")
    ax.set_title("EMERREL — ANN")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    if emerac_raw[-1] > 0:
        ax.plot(dias, emerac_raw / emerac_raw[-1], label="EMERAC cruda (norm.)", color="orange", alpha=0.5)
    else:
        ax.plot(dias, emerac_raw, label="EMERAC cruda", color="orange", alpha=0.5)
    if emerac[-1] > 0:
        ax.plot(dias, emerac / emerac[-1], label="EMERAC procesada (norm.)", color="green", linewidth=2)
    else:
        ax.plot(dias, emerac, label="EMERAC procesada", color="green", linewidth=2)
    ax.set_xlabel("Día Juliano")
    ax.set_ylabel("EMERAC (0–1 relativo al período)")
    ax.set_title("EMERAC — ANN")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)


# ===============================================================
# CLASIFICACIÓN COMPLETA
# ===============================================================

st.subheader("🌱 Clasificación del patrón (ANN + centroides + reglas)")

C, labels_df, rep_year, curvas_hist = safe(
    lambda: load_centroides_y_historia(),
    "Error cargando centroides e histórico de EMERAC"
)

if C is None:
    st.stop()

patron, vals, dists, probs = clasificar_patron_desde_ann(dias, emerac, C)

if patron is None:
    st.error("No se pudo clasificar el patrón (la curva ANN no alcanza suficiente emergencia).")
    st.stop()

st.markdown(f"### 🟢 Patrón resultante: **{patron}**")
st.write("**Probabilidades relativas por patrón:**")
st.json(probs)

dist_table = pd.DataFrame({
    "Patrón": list(C.index),
    "Distancia al centroide": dists,
    "Probabilidad relativa": [probs.get(str(p), np.nan) for p in C.index]
}).sort_values("Distancia al centroide")

st.dataframe(dist_table, use_container_width=True)

if str(patron) in rep_year:
    st.info(f"Año histórico representativo del patrón **{patron}**: **{rep_year[str(patron)]}**")


# ===============================================================
# 🔎 DIAGNÓSTICO VISUAL AVANZADO
# ===============================================================

st.subheader("🔎 Diagnóstico visual avanzado: patrones vs curva ANN")

if (labels_df is None) or (curvas_hist is None):
    st.info("No hay histórico suficiente para generar el diagnóstico visual avanzado.")
else:
    try:
        jd_min = int(max(1, np.floor(dias.min())))
        jd_max = int(min(365, np.ceil(dias.max())))
        stats = build_patron_stats(C, labels_df, curvas_hist, jd_min=jd_min, jd_max=jd_max)

        # Curva ANN normalizada e interpolada a la grilla común
        grid = np.arange(jd_min, jd_max + 1)
        if emerac[-1] > 0:
            emerac_norm = emerac / emerac[-1]
        else:
            emerac_norm = emerac
        emerac_ann_interp = np.interp(grid, dias, emerac_norm)

        colors = {"Early": "green", "Intermediate": "gold", "Extended": "red", "Late": "blue"}

        fig_diag, axd = plt.subplots(figsize=(12, 6))
        for pat, s in stats.items():
            g = s["grid"]
            axd.fill_between(g, s["p25"], s["p75"],
                             color=colors.get(pat, "gray"), alpha=0.15)
            axd.plot(g, s["median"],
                     color=colors.get(pat, "gray"), lw=2,
                     label=f"{pat} — Mediana")

        axd.plot(grid, emerac_ann_interp,
                 color="black", lw=3, label="Curva ANN evaluada")

        # Percentiles ANN (JD25–95) como líneas verticales
        if vals is not None:
            jd25, jd50, jd75, jd95 = vals
            for x, lbl in zip([jd25, jd50, jd75, jd95], ["JD25", "JD50", "JD75", "JD95"]):
                axd.axvline(x, color="black", linestyle="--", alpha=0.4)
                axd.text(x, 1.02, lbl, rotation=90,
                         va="bottom", ha="center", fontsize=8)

        axd.set_xlabel("Día Juliano")
        axd.set_ylabel("Emergencia acumulada (normalizada)")
        axd.set_title("Gráfico diagnóstico — Curva ANN vs patrones históricos (bandas 25–75%)")
        axd.legend(loc="lower right")
        axd.grid(alpha=0.2)
        fig_diag.tight_layout()
        st.pyplot(fig_diag)

    except Exception as e:
        st.error(f"No se pudo generar el diagnóstico visual avanzado: {e}")


# ===============================================================
# DESCARGA DE SERIE ANN
# ===============================================================

st.download_button(
    "📥 Descargar EMERREL/EMERAC simulada (ANN)",
    df_ann.to_csv(index=False).encode("utf-8"),
    "emergencia_simulada_ANN_v85.csv",
    mime="text/csv"
)



