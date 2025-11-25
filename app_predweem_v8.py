# ===============================================================
# 🌾 PREDWEEM v8.4 — AVEFA Predictor 2026 
# ANN + Centroides + Reglas agronómicas EXTENDIDAS
# ===============================================================
# REGLAS AGRONÓMICAS:
# 1) LATE:
#    - inicio emergencia (JD_ini) > 85
#    - JD50 > 150
#    Si falla → prohibido.
#
# 2) EXTENDED (prioridad absoluta):
#    - JD_ini entre 50 y 80 
#    - JD50 > 150
#    Si cumple → SE FUERZA EXTENDED.
#
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
st.set_page_config(page_title="PREDWEEM v8.4 — AVEFA 2026", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 PREDWEEM v8.4 — AVEFA Predictor 2026")
st.subheader("ANN + Centroides + Reglas agronómicas extendidas")

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

# ---------------------------------------------------------
# GRÁFICOS
# ---------------------------------------------------------
def plot_comparativo_curvas(jd, emerac_pred, emerac_obs, nombre_obs="Observada"):
    fig, ax = plt.subplots(figsize=(10, 5))

    pred_norm = emerac_pred / emerac_pred.max() if emerac_pred.max() > 0 else emerac_pred
    obs_norm  = emerac_obs / emerac_obs.max()   if emerac_obs.max() > 0 else emerac_obs

    ax.plot(jd, pred_norm, color="blue", linewidth=3, label="Predicha (ANN)")
    ax.plot(jd, obs_norm, color="red", linestyle="--", linewidth=2, label=nombre_obs)

    ax.set_xlabel("Día Juliano")
    ax.set_ylabel("Emergencia acumulada normalizada (0–1)")
    ax.set_title("Comparación — ANN vs Observada")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig

def plot_comparativo_visual(jd, emerac_pred, emerac_obs,
                            perc_pred=None, perc_obs=None,
                            nombre_obs="Observada"):
    fig, ax = plt.subplots(figsize=(12, 6))

    pred = emerac_pred / emerac_pred.max() if emerac_pred.max() > 0 else emerac_pred
    obs  = emerac_obs / emerac_obs.max()   if emerac_obs.max() > 0 else emerac_obs

    ax.plot(jd, pred, color="blue", linewidth=3, label="Predicha (ANN)")
    ax.plot(jd, obs, color="red", linestyle="--", linewidth=2, label=nombre_obs)
    ax.fill_between(jd, pred, obs, color="gray", alpha=0.25)

    if perc_pred is not None:
        for p, c in zip(perc_pred, ["#0033aa", "#0044dd", "#0055ff", "#0077ff"]):
            ax.axvline(p, color=c, linestyle="-", alpha=0.7)

    if perc_obs is not None:
        for p, c in zip(perc_obs, ["#aa0000", "#cc0000", "#ee0000", "#ff2222"]):
            ax.axvline(p, color=c, linestyle="--", alpha=0.8)

    ax.grid(alpha=0.25)
    ax.legend()
    return fig

# ===============================================================
# PERCENTILES JD25–95
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
    return np.array([np.interp(q, y, jd) for q in qs], dtype=float)

# ===============================================================
# CARGA HISTÓRICO + CENTROIDES
# ===============================================================

def _load_curves_emereac():
    curvas = {}

    # 1977–1998
    try:
        xls1 = pd.ExcelFile(BASE / "emergencia_acumulada_interpolada 1977-1998.xlsx")
        for sh in xls1.sheet_names:
            df = pd.read_excel(xls1, sheet_name=sh)
            year = int(sh.split("_")[-1])
            curvas[year] = df[["JD", "EMERAC"]].copy()
    except:
        pass

    # 2000–2015
    try:
        xls2 = pd.ExcelFile(BASE / "emergencia_2000_2015_interpolada.xlsx")
        for sh in xls2.sheet_names:
            df = pd.read_excel(xls2, sheet_name=sh)
            year = int(sh.split("_")[-1])
            curvas[year] = df[["JD", "EMERAC"]].copy()
    except:
        pass

    return curvas

def _assign_labels_from_centroids(curvas, C):
    registros = []
    for year, df in curvas.items():
        vals = _compute_jd_percentiles(df["JD"], df["EMERAC"])
        if vals is None:
            continue
        d25, d50, d75, d95 = vals
        v = np.array([d25, d50, d75, d95])
        dists = np.linalg.norm(C.values - v, axis=1)
        patron = C.index[np.argmin(dists)]

        registros.append({
            "anio": year,
            "patron": str(patron),
            "JD25": d25, "JD50": d50, "JD75": d75, "JD95": d95
        })

    return pd.DataFrame(registros)

@st.cache_resource
def load_centroides_y_historia():
    cent = joblib.load(BASE / "predweem_model_centroides.pkl")
    C = cent["centroides"]
    curvas = _load_curves_emereac()
    labels_df = _assign_labels_from_centroids(curvas, C)

    rep_year = {}
    for patron in C.index:
        sub = labels_df[labels_df["patron"] == str(patron)]
        if not sub.empty:
            vc = C.loc[patron][["JD25", "JD50", "JD75", "JD95"]].values
            M = sub[["JD25", "JD50", "JD75", "JD95"]].values
            best = np.argmin(np.linalg.norm(M - vc, axis=1))
            rep_year[str(patron)] = int(sub.iloc[best]["anio"])

    return C, labels_df, rep_year, curvas

# ===============================================================
# ANN
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
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

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
    IW  = np.load(BASE / "IW.npy")
    bIW = np.load(BASE / "bias_IW.npy")
    LW  = np.load(BASE / "LW.npy")
    bLW = np.load(BASE / "bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)

def postprocess_emergence(emerrel_raw, smooth=True, window=3, clip_zero=True):
    emer = np.maximum(emerrel_raw, 0.0) if clip_zero else emerrel_raw
    if smooth and window > 1:
        k = np.ones(int(window)) / window
        emer = np.convolve(emer, k, mode="same")
    emerac = np.cumsum(emer)
    return emer, emerac

# ===============================================================
# 🔥 CLASIFICACIÓN CON REGLAS AGRONÓMICAS EXTENDIDAS
# ===============================================================

def clasificar_patron_desde_ann(dias, emerac, C):
    """
    Reglas:
    LATE:
        - JD_ini > 85
        - JD50 > 150
    EXTENDED (PRIORIDAD ABSOLUTA):
        - JD_ini entre 50 y 80
        - JD50 > 150
    """

    vals = _compute_jd_percentiles(dias, emerac)
    if vals is None:
        return None, None, None, None

    d25, d50, d75, d95 = vals
    v = np.array([d25, d50, d75, d95])

    dists = np.linalg.norm(C.values - v, axis=1)

    w = 1.0 / (dists + 1e-6)
    p = w / w.sum()
    prob_dict = {str(C.index[i]): float(p[i]) for i in range(len(C.index))}

    emerac = np.asarray(emerac)
    dias = np.asarray(dias)

    # JD_ini
    idx = np.where(emerac > 0.01)[0]
    JD_ini = dias[idx[0]] if len(idx) > 0 else np.inf

    JD50 = d50

    # ------------------------------------------
    # EXTENDED (prioridad absoluta)
    # ------------------------------------------
    if (50 <= JD_ini <= 80) and (JD50 > 150) and ("Extended" in C.index):
        patron = "Extended"
        idx_ext = list(C.index).index("Extended")

        d_mod = dists.copy()
        d_mod[idx_ext] = -1.0

        w = 1.0 / (d_mod - d_mod.min() + 1e-6)
        p = w / w.sum()
        prob_dict = {str(C.index[i]): float(p[i]) for i in range(len(C.index))}

        return patron, vals, d_mod, prob_dict

    # ------------------------------------------
    # LATE (descartar si no cumple)
    # ------------------------------------------
    permitir_late = True
    if JD_ini <= 85:
        permitir_late = False
    if JD50 <= 150:
        permitir_late = False

    if (not permitir_late) and ("Late" in C.index):
        dists[list(C.index).index("Late")] = np.inf

        w = 1.0 / (dists + 1e-6)
        p = w / w.sum()
        prob_dict = {str(C.index[i]): float(p[i]) for i in range(len(C.index))}

    idx_min = np.argmin(dists)
    patron = str(C.index[idx_min])

    return patron, vals, dists, prob_dict

# ===============================================================
# DETECCIÓN COLUMNAS METEO
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

    c_jd = pick("jd", "julian", "dia")
    c_tmax = pick("tmax")
    c_tmin = pick("tmin")
    c_prec = pick("prec", "lluvia")

    if None in (c_jd, c_tmax, c_tmin, c_prec):
        raise ValueError("No se identificaron JD/TMAX/TMIN/Prec.")

    df["JD"] = pd.to_numeric(df[c_jd], errors="coerce")
    df["TMAX"] = pd.to_numeric(df[c_tmax], errors="coerce")
    df["TMIN"] = pd.to_numeric(df[c_tmin], errors="coerce")
    df["Prec"] = pd.to_numeric(df[c_prec], errors="coerce")

    if pick("fecha"):
        df["Fecha"] = pd.to_datetime(df[pick("fecha")], errors="coerce")
    else:
        df["Fecha"] = None

    df = df.dropna(subset=["JD"])
    return df.sort_values("JD")

# ===============================================================
# SIDEBAR CONTROLES
# ===============================================================

with st.sidebar:
    st.header("Ajustes ANN")
    use_smoothing = st.checkbox("Suavizar EMERREL", True)
    window_size = st.slider("Ventana suavizado", 1, 9, 3)
    clip_zero = st.checkbox("Cortar negativos", True)

# ===============================================================
# CARGA ARCHIVO METEOROLÓGICO
# ===============================================================
st.subheader("📤 Cargar archivo meteorológico")
uploaded = st.file_uploader("CSV/XLSX", type=["csv", "xlsx"])

modelo_ann = safe(load_ann, "Error cargando ANN")

if uploaded is None:
    st.info("Cargar archivo meteorológico para iniciar.")
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

st.success("Archivo cargado correctamente.")
st.dataframe(df_raw, use_container_width=True)

try:
    df_meteo = normalizar_meteo(df_raw)
except Exception as e:
    st.error(f"Error normalizando archivo: {e}")
    st.stop()

if modelo_ann is None:
    st.stop()

# ===============================================================
# ANN
# ===============================================================
st.subheader("🔍 ANN → EMERREL / EMERAC")

df_ann = df_meteo.copy().sort_values("JD")
dias = df_ann["JD"].to_numpy()

X = df_ann[["JD", "TMAX", "TMIN", "Prec"]].to_numpy()

emerrel_raw, emerac_raw = modelo_ann.predict(X)
emerrel, emerac = postprocess_emergence(
    emerrel_raw, smooth=use_smoothing, window=window_size, clip_zero=clip_zero
)

df_ann["EMERREL"] = emerrel
df_ann["EMERAC"] = emerac

# ---------------------------------------------------------
# Gráficos ANN
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(dias, emerrel_raw, label="Cruda", color="red", alpha=0.5)
    ax.plot(dias, emerrel, label="Procesada", color="blue", linewidth=2)
    ax.legend()
    ax.set_title("EMERREL ANN")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(dias, emerac / emerac.max(), label="Procesada", color="green")
    ax.plot(dias, emerac_raw / emerac_raw.max(), label="Cruda", color="orange", alpha=0.5)
    ax.legend()
    ax.set_title("EMERAC ANN")
    st.pyplot(fig)

# ===============================================================
# CLASIFICACIÓN — ANN + CENTROIDES + REGLAS AGRONÓMICAS
# ===============================================================

st.subheader("🌱 Clasificación del patrón (ANN + centroides + reglas)")

C, labels_df, rep_year, curvas_hist = safe(
    lambda: load_centroides_y_historia(),
    "No se pudieron cargar centroides/histórico"
)

if C is None:
    st.stop()

patron_ann, vals_ann, dists_ann, prob_dict = clasificar_patron_desde_ann(dias, emerac, C)

if patron_ann is None:
    st.error("No se pudo clasificar el patrón.")
    st.stop()

st.markdown(f"### 🟢 Patrón seleccionado: **{patron_ann}**")

st.write("**Probabilidades relativas:**")
st.json(prob_dict)

# Distancias tabla
dist_table = pd.DataFrame({
    "patron": list(C.index),
    "distancia": dists_ann,
    "prob": [prob_dict[str(p)] for p in C.index]
}).sort_values("distancia")

st.dataframe(dist_table, use_container_width=True)

# Año representativo
if str(patron_ann) in rep_year:
    st.success(f"Año representativo del patrón {patron_ann}: **{rep_year[str(patron_ann)]}**")
else:
    st.info("No hay año representativo disponible.")

# ===============================================================
# COMPARACIÓN CON OBSERVADA
# ===============================================================
st.subheader("📊 Comparación con curva observada (opcional)")

archivo_obs = st.file_uploader("Cargar curva observada", key="obs", type=["csv", "xlsx"])

if archivo_obs:
    df_obs = pd.read_csv(archivo_obs) if archivo_obs.name.endswith(".csv") else pd.read_excel(archivo_obs)

    col_jd = [c for c in df_obs.columns if "jd" in c.lower()][0]
    col_emer = [c for c in df_obs.columns if "emerac" in c.lower() or "emerrel" in c.lower()][0]

    jd_obs = pd.to_numeric(df_obs[col_jd], errors="coerce")
    val_obs = pd.to_numeric(df_obs[col_emer], errors="coerce")

    if "rel" in col_emer.lower():
        emerac_obs = np.cumsum(np.maximum(val_obs, 0))
    else:
        emerac_obs = val_obs

    emerac_obs_interp = np.interp(dias, jd_obs, emerac_obs)

    fig_c = plot_comparativo_curvas(dias, emerac, emerac_obs_interp)
    st.pyplot(fig_c)

    fig_c2 = plot_comparativo_visual(
        dias, emerac, emerac_obs_interp,
        perc_pred=_compute_jd_percentiles(dias, emerac),
        perc_obs=_compute_jd_percentiles(dias, emerac_obs_interp)
    )
    st.pyplot(fig_c2)

# ===============================================================
# RADAR JD25–95
# ===============================================================
st.subheader("🎯 Radar percentiles JD25–95")

if vals_ann is not None:
    labels = ["JD25", "JD50", "JD75", "JD95"]
    vals_pat = C.loc[patron_ann][labels].to_numpy()

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ax.plot(angles, np.concatenate((vals_ann, [vals_ann[0]])), label="Año ANN")
    ax.plot(angles, np.concatenate((vals_pat, [vals_pat[0]])), label=f"Patrón {patron_ann}")

    ax.fill(angles, np.concatenate((vals_ann, [vals_ann[0]])), alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend()
    st.pyplot(fig)

# ===============================================================
# COMPARACIÓN CON AÑO REPRESENTATIVO
# ===============================================================
st.subheader("📊 Comparación con año representativo")

if patron_ann in rep_year:
    yr = rep_year[patron_ann]
    if yr in curvas_hist:
        df_r = curvas_hist[yr]
        em_rep = np.interp(dias, df_r["JD"], df_r["EMERAC"])

        fig_r = plot_comparativo_curvas(dias, emerac, em_rep, nombre_obs=f"Año {yr}")
        st.pyplot(fig_r)

# ===============================================================
# CERTEZA DIARIA
# ===============================================================
st.subheader("📈 Certeza diaria del patrón")

jd_eval = []
probs_sel = []

for i in range(4, len(dias)):
    jd_sub = dias[:i+1]
    emerac_sub = emerac[:i+1]
    vals_i = _compute_jd_percentiles(jd_sub, emerac_sub)
    if vals_i is None:
        continue

    v = np.array(vals_i)
    dists = np.linalg.norm(C.values - v, axis=1)
    w = 1.0/(dists + 1e-6)
    p = w / w.sum()

    probs_sel.append(float(p[list(C.index).index(patron_ann)]))
    jd_eval.append(jd_sub[-1])

if len(jd_eval) > 0:
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(jd_eval, probs_sel, color="green", lw=2)
    ax.set_ylim(0,1)
    ax.set_title("Evolución diaria de certeza")
    ax.set_xlabel("JD")
    ax.set_ylabel(f"P({patron_ann})")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ===============================================================
# DESCARGA
# ===============================================================
st.download_button(
    "📥 Descargar EMERREL/EMERAC simulada",
    df_ann.to_csv(index=False).encode("utf-8"),
    "emergencia_simulada_ANN.csv",
    mime="text/csv"
)












