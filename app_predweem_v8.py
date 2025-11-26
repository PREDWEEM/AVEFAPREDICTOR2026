# ===============================================================
# 🌾 PREDWEEM v8.5 — AVEFA Predictor 2026
# ANN + Centroides + Reglas agronómicas avanzadas
# + Diagnóstico visual avanzado
# + Radar JD25–JD95
# + Diagnóstico agronómico simple + mixto
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
st.subheader("ANN + Centroides + Reglas fisiológicas avanzadas + Diagnóstico agronómico")

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
    y_obs = np.asarray(y_obs, float)
    return float(np.sqrt(np.mean((y_pred - y_obs) ** 2)))


# ===============================================================
# COMPUTO DE PERCENTILES (JD25–JD95)
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
    # 1977–1998
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

    # 2000–2015
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
    registros = []
    for year, df in curvas.items():
        vals = _compute_jd_percentiles(df["JD"], df["EMERAC"])
        if vals is None:
            continue
        d25, d50, d75, d95 = vals
        v = np.array([d25, d50, d75, d95], float)
        dists = np.linalg.norm(C.values - v, axis=1)
        patron = C.index[np.argmin(dists)]
        registros.append({
            "anio": int(year),
            "patron": str(patron),
            "JD25": d25,
            "JD50": d50,
            "JD75": d75,
            "JD95": d95
        })
    if len(registros) == 0:
        return pd.DataFrame(columns=["anio", "patron", "JD25", "JD50", "JD75", "JD95"])
    return pd.DataFrame(registros)


@st.cache_resource
def load_centroides_y_historia():
    cent = joblib.load(BASE / "predweem_model_centroides.pkl")
    C = cent["centroides"]  # DataFrame con columnas JD25–JD95, índice = patrones
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
    IW = np.load(BASE / "IW.npy")
    bIW = np.load(BASE / "bias_IW.npy")
    LW = np.load(BASE / "LW.npy")
    bLW = np.load(BASE / "bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)


def postprocess_emergence(raw, smooth=True, window=3, clip_zero=True):
    emer = np.maximum(raw, 0.0) if clip_zero else raw
    if smooth and window > 1:
        window = int(window)
        kernel = np.ones(window) / window
        emer = np.convolve(emer, kernel, mode="same")
    emerac = np.cumsum(emer)
    return emer, emerac


# ===============================================================
# REGLAS AGRONÓMICAS (EARLY / INTERMEDIATE / EXTENDED / LATE)
# ===============================================================
def aplicar_reglas_agronomicas(JD_ini, JD25, JD50, JD75, JD95, patron_inicial=None):
    """
    Devuelve patrón 'override' si las reglas agronómicas son claras.
    Si no hay override, devuelve None y se usa el centroide más cercano.
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
    Devuelve:
      patron_final, vals(JD25–JD95), distancias a centroides, prob_dict
    """

    vals = _compute_jd_percentiles(dias, emerac)
    if vals is None:
        return None, None, None, None

    d25, d50, d75, d95 = vals
    v = np.array([d25, d50, d75, d95], float)

    dists = np.linalg.norm(C.values - v, axis=1)
    w = 1.0 / (dists + 1e-6)
    p = w / w.sum()
    prob_dict = {str(C.index[i]): float(p[i]) for i in range(len(C.index))}

    emerac = np.asarray(emerac, float)
    dias = np.asarray(dias, float)

    idx = np.where(emerac > 0.01)[0]
    JD_ini = dias[idx[0]] if len(idx) > 0 else np.inf

    # Aplicar reglas agronómicas
    override = aplicar_reglas_agronomicas(JD_ini, d25, d50, d75, d95, None)

    if override is not None and override in C.index:
        patron_final = override
        d_mod = dists.copy()
        idxp = list(C.index).index(patron_final)
        d_mod[idxp] = -1.0  # forzamos este patrón

        w = 1.0 / (d_mod - d_mod.min() + 1e-6)
        p = w / w.sum()
        prob_mod = {str(C.index[i]): float(p[i]) for i in range(len(C.index))}
        return patron_final, vals, d_mod, prob_mod

    # Si no hay override → centroide más cercano
    idx_min = int(np.argmin(dists))
    patron = str(C.index[idx_min])
    return patron, vals, dists, prob_dict


# ===============================================================
# NORMALIZAR ARCHIVO METEOROLÓGICO
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
    c_prec = pick("prec", "lluv")
    c_fecha = pick("fecha")

    if None in (c_jd, c_tmax, c_tmin, c_prec):
        raise ValueError("No se identifican JD / TMAX / TMIN / Prec en el archivo.")

    df["JD"] = pd.to_numeric(df[c_jd], errors="coerce")
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
# ESTADÍSTICAS PARA DIAGNÓSTICO VISUAL (BANDAS 25–75)
# ===============================================================
def build_patron_stats(C, labels_df, curvas_hist, jd_min=1, jd_max=365):
    """
    Para cada patrón genera curvas:
      - mediana
      - p25
      - p75
    en una grilla común de JD.
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
            "p25": np.nanpercentile(A, 25, axis=0),
            "p75": np.nanpercentile(A, 75, axis=0),
        }
    return stats


# ===============================================================
# SIDEBAR
# ===============================================================
with st.sidebar:
    st.header("Ajustes ANN")
    smooth = st.checkbox("Suavizar EMERREL", True)
    window = st.slider("Ventana de suavizado (días)", 1, 9, 3)
    clip = st.checkbox("Recortar negativos a 0", True)

# ===============================================================
# CARGA METEOROLÓGICA (Automática + Manual)
# ===============================================================
st.subheader("📤 Datos Meteorológicos (Automático desde meteo_daily.csv o Manual)")

def cargar_meteo_daily_csv():
    """Carga automática desde meteo_daily.csv si el archivo existe."""
    fname = BASE / "meteo_daily.csv"
    if fname.exists():
        try:
            df = pd.read_csv(fname)
            st.success("📌 Datos meteorológicos cargados automáticamente desde **meteo_daily.csv**")
            return df
        except Exception as e:
            st.error(f"Error leyendo meteo_daily.csv: {e}")
            return None
    return None


# ---------- PRIORIDAD 1: archivo subido manualmente ----------
uploaded = st.file_uploader("Subir archivo meteorológico (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded is not None:
    st.info("Usando archivo meteorológico subido manualmente.")
    try:
        df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.success("Archivo meteorológico cargado correctamente.")
    except Exception as e:
        st.error(f"Error leyendo el archivo subido: {e}")
        st.stop()

else:
    # ---------- PRIORIDAD 2: meteo_daily.csv automático ----------
    df_auto = cargar_meteo_daily_csv()
    if df_auto is not None:
        df_raw = df_auto
    else:
        st.warning("⚠ No se subió archivo y no existe meteo_daily.csv.")
        st.info("Suba un archivo meteorológico para continuar.")
        st.stop()


# ---------- Normalización ----------
st.dataframe(df_raw, use_container_width=True)

try:
    df_meteo = normalizar_meteo(df_raw)
except Exception as e:
    st.error(f"Error normalizando el archivo meteorológico: {e}")
    st.stop()

# Si ANN no se cargó → stop
modelo_ann = safe(load_ann, "Error cargando pesos de la ANN")
if modelo_ann is None:
    st.stop()


# ===============================================================
# ANN → EMERREL / EMERAC
# ===============================================================
st.subheader("🔍 Emergencia simulada por ANN (EMERREL / EMERAC)")

df_ann = df_meteo.copy()
dias = df_ann["JD"].to_numpy(float)
X = df_ann[["JD", "TMAX", "TMIN", "Prec"]].to_numpy(float)

emerrel_raw, emerac_raw = modelo_ann.predict(X)
emerrel, emerac = postprocess_emergence(emerrel_raw, smooth=smooth, window=window, clip_zero=clip)

df_ann["EMERREL"] = emerrel
df_ann["EMERAC"] = emerac

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(dias, emerrel_raw, label="EMERREL cruda", color="red", alpha=0.5)
    ax.plot(dias, emerrel, label="EMERREL procesada", color="blue", linewidth=2)
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
# CLASIFICACIÓN DEL PATRÓN
# ===============================================================
st.subheader("🌱 Clasificación del patrón (ANN + centroides + reglas)")

C, labels_df, rep_year, curvas_hist = safe(
    lambda: load_centroides_y_historia(),
    "Error cargando centroides e histórico de EMERAC"
)

if C is None:
    st.stop()

patron, vals, dists, prob_dict = clasificar_patron_desde_ann(dias, emerac, C)

if patron is None:
    st.error("No se pudo clasificar el patrón (la curva ANN no alcanza suficiente emergencia).")
    st.stop()

st.markdown(f"### 🟢 Patrón resultante: **{patron}**")
st.write("**Probabilidades relativas por patrón:**")
st.json(prob_dict)

dist_table = pd.DataFrame({
    "Patrón": [str(p) for p in C.index],
    "Distancia al centroide": dists,
    "Probabilidad relativa": [prob_dict.get(str(p), np.nan) for p in C.index]
}).sort_values("Distancia al centroide")

st.dataframe(dist_table, use_container_width=True)

if str(patron) in rep_year:
    st.info(f"Año histórico representativo del patrón **{patron}**: **{rep_year[str(patron)]}**")


# ===============================================================
# RADAR JD25–JD95 (ANN vs CENTROIDES)
# ===============================================================
st.subheader("📈 Radar de percentiles (JD25–JD95)")

try:
    if vals is None:
        st.warning("No hay percentiles válidos para trazar el radar.")
    else:
        jd25_ann, jd50_ann, jd75_ann, jd95_ann = vals
        ann_vec = np.array([jd25_ann, jd50_ann, jd75_ann, jd95_ann], float)

        patrones = list(C.index)
        centroid_matrix = C[["JD25", "JD50", "JD75", "JD95"]].to_numpy(float)

        labels_radar = ["JD25", "JD50", "JD75", "JD95"]
        angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False)
        angles = np.concatenate([angles, angles[:1]])

        fig_rad, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)

        color_map = {"Early": "green", "Intermediate": "gold",
                     "Extended": "red", "Late": "blue"}

        for i, pat in enumerate(patrones):
            vec = centroid_matrix[i]
            vec = np.concatenate([vec, vec[:1]])
            ax.plot(angles, vec, lw=2, label=f"{pat} (centroide)",
                    color=color_map.get(pat, "gray"), alpha=0.8)
            ax.fill(angles, vec, alpha=0.08, color=color_map.get(pat, "gray"))

        ann_plot = np.concatenate([ann_vec, ann_vec[:1]])
        ax.plot(angles, ann_plot, lw=3, color="black", label="Año evaluado (ANN)")
        ax.scatter(angles, ann_plot, color="black")

        if patron in rep_year:
            yr = rep_year[patron]
            if yr in curvas_hist:
                dfp = curvas_hist[yr]
                pvals = _compute_jd_percentiles(dfp["JD"], dfp["EMERAC"])
                if pvals is not None:
                    rep = np.concatenate([pvals, pvals[:1]])
                    ax.plot(angles, rep, lw=2, color=color_map.get(patron, "gray"),
                            linestyle="--", label=f"Representativo {patron} ({yr})")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_radar, fontsize=12)
        ax.set_rlabel_position(0)
        ax.grid(True, alpha=0.3)
        ax.set_title("Radar JD25–JD95 — Comparación ANN vs patrones", fontsize=14)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        st.pyplot(fig_rad)

except Exception as e:
    st.error(f"No se pudo generar el radar JD25–95: {e}")


# ===============================================================
# DIAGNÓSTICO VISUAL AVANZADO (BANDAS 25–75 + CURVA ANN)
# ===============================================================
st.subheader("🔎 Diagnóstico visual avanzado: patrones vs curva ANN")

if (labels_df is None) or (curvas_hist is None):
    st.info("No hay histórico suficiente para generar el diagnóstico visual avanzado.")
else:
    try:
        jd_min = int(max(1, np.floor(dias.min())))
        jd_max = int(min(365, np.ceil(dias.max())))
        stats = build_patron_stats(C, labels_df, curvas_hist, jd_min=jd_min, jd_max=jd_max)

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
# ANÁLISIS AGRONÓMICO INTELIGENTE (PATRÓN DOMINANTE)
# ===============================================================
st.subheader("🌾 Análisis Agronómico Inteligente del patrón dominante")

if patron is None:
    st.info("Aún no hay patrón asignado.")
else:
    st.markdown(f"### 🟢 Patrón asignado: **{patron}**")

    analisis = {
        "Early": """
### 🟩 EARLY (Temprano)
- Inicio: JD 45–75
- JD50 < 110
- Emergencia muy compacta y agresiva al inicio del ciclo.
- El barbecho es la variable más crítica: fallas → explosión temprana.
- Requiere residuales potentes pre-siembra y pre-emergencia (Flumioxazin, Sulfentrazone, Metribuzin).
- Manejo post-emergente suele llegar tarde frente al pico de emergencia.
""",
        "Intermediate": """
### 🟨 INTERMEDIATE (Intermedio)
- Inicio: JD 70–100
- JD50 ≈ 120–140
- Emergencia en 2–3 oleadas, más distribuida en el tiempo.
- Requiere combinación de residuales + monitoreo frecuente.
- Residuales de persistencia media funcionan bien, ajustando post-emergente entre JD 100–150.
""",
        "Extended": """
### 🟥 EXTENDED (Extendido)
- Inicio: JD 50–80
- JD50 > 150
- Emergencia prolongada: múltiples cohortes tempranas y tardías.
- Situación de alta presión: exige residuales prolongados y solapados.
- Necesario monitoreo cada 12–14 días para evitar escapes.
""",
        "Late": """
### 🟦 LATE (Tardío)
- Inicio: >JD 85
- JD50 > 150
- Emergencia tardía asociada a lluvias y calor de primavera.
- Residuales pre-siembra pueden quedar cortos.
- Manejo óptimo con graminicidas post-emergentes (1–3 macollos) + residual pos-siembra.
"""
    }

    st.markdown(analisis.get(patron, "No hay análisis disponible para este patrón."))

    st.markdown("### ⚠ Alertas de manejo")
    if patron == "Early":
        st.warning("⚠ Alta presión temprana: residuales pre-siembra son críticos.")
    elif patron == "Intermediate":
        st.info("ℹ Atención a oleadas múltiples: ajustar ventana del post-emergente.")
    elif patron == "Extended":
        st.error("❗ Riesgo alto de escapes: se recomienda solapamiento de residuales.")
    elif patron == "Late":
        st.warning("⚠ Emergencia tardía intensa: no retrasar aplicaciones post-emergentes.")


# ===============================================================
# DIAGNÓSTICO AGRONÓMICO MIXTO (PATRONES MÁS PROBABLES)
# ===============================================================
st.subheader("🌾 Diagnóstico agronómico combinado según patrones probables")

if prob_dict is None:
    st.info("Aún no se calcularon probabilidades.")
else:
    df_probs = pd.DataFrame([
        {"Patrón": p, "Probabilidad": round(prob_dict[p], 3)}
        for p in sorted(prob_dict, key=prob_dict.get, reverse=True)
    ])
    st.markdown("### 🔢 Probabilidades relativas por patrón")
    st.dataframe(df_probs, use_container_width=True)

    top = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    patron1, p1 = top[0]
    patron2, p2 = top[1] if len(top) > 1 else (None, 0.0)
    patron3, p3 = top[2] if len(top) > 2 else (None, 0.0)

    st.markdown(f"### 🥇 Patrón dominante: **{patron1}** ({p1:.1%})")
    if patron2:
        st.markdown(f"### 🥈 Segundo patrón probable: **{patron2}** ({p2:.1%})")
    if patron3:
        st.markdown(f"### 🥉 Tercero: **{patron3}** ({p3:.1%})")

    desc = {
        "Early": {
            "inicio": "Muy temprano (JD 45–75)",
            "agresividad": "Muy alta al inicio del cultivo",
            "duracion": "Corta y compacta",
            "manejo": "Residuales potentes pre-siembra + pre-emergencia",
            "riesgos": "Fallas en barbecho → explosión temprana"
        },
        "Intermediate": {
            "inicio": "Moderado (JD 70–100)",
            "agresividad": "Media, distribuida",
            "duracion": "2–3 oleadas",
            "manejo": "Residual + post-emergente en ventana JD 100–150",
            "riesgos": "Oleadas tardías si no se mantiene cobertura"
        },
        "Extended": {
            "inicio": "Variable (JD 50–80)",
            "agresividad": "Alta por emergencia prolongada",
            "duracion": "Extensa (JD25–JD95 muy separados)",
            "manejo": "Solapamiento de residuales + monitoreo 12–14 días",
            "riesgos": "Escapes tardíos, competencia sostenida"
        },
        "Late": {
            "inicio": "Tardío (JD >85)",
            "agresividad": "Alta en el pico tardío (JD 160–230)",
            "duracion": "Media, con pico abrupto",
            "manejo": "Post-emergentes (1–3 macollos) + residual pos-siembra",
            "riesgos": "Explosión tardía si se atrasa el post-emergente"
        }
    }

    st.markdown("## 🌱 Síntesis agronómica combinada (Top 3 patrones)")
    def combinar(p, peso):
        d = desc[p]
        txt = (
            f"### **{p}** ({peso:.1%})\n"
            f"- Inicio: {d['inicio']}\n"
            f"- Agresividad: {d['agresividad']}\n"
            f"- Duración: {d['duracion']}\n"
            f"- Manejo recomendado: {d['manejo']}\n"
            f"- Riesgos: {d['riesgos']}\n"
        )
        return txt

    for p, peso in top[:3]:
        st.markdown(combinar(p, peso))

    st.markdown("## 🧠 Diagnóstico Agronómico Integrado (AAI MIXTO)")

    diag = ""
    if patron1 == "Early" and patron2 == "Extended":
        diag = """
### 🟥🟩 Sistema mixto Early + Extended
- Emergencia muy temprana y sostenida → máxima presión del sistema.
- Requiere residuales potentes y solapados.
- Monitoreo cada 12 días para capturar oleadas largas.
- Post-emergente necesario en ventana temprana **y** tardía.
"""
    elif patron1 == "Extended" and patron2 == "Late":
        diag = """
### 🟥🟦 Sistema tardío y prolongado (Extended + Late)
- Inicios variables pero con pico fuerte tardío.
- Residuales pre-siembra pueden quedar cortos.
- Recomendado: residual pos-siembra + post-emergentes en 1–3 macollos.
- Riesgo alto de explosión tardía si se atrasa el post-emergente.
"""
    elif patron1 == "Early" and patron2 == "Intermediate":
        diag = """
### 🟩🟨 Sistema temprano con oleadas medias
- Foco principal: barbecho + residual temprano.
- Ajustar post-emergente entre JD 110–150.
- Riesgo: primera cohorte muy agresiva si el barbecho es deficiente.
"""
    elif patron1 == "Intermediate" and patron2 == "Extended":
        diag = """
### 🟨🟥 Sistema de carga media–alta prolongada
- Emergencia en varias oleadas + prolongada.
- Residuales de persistencia media pueden ser insuficientes.
- Reforzar persistencia y monitoreo para evitar escapes tardíos.
"""
    elif patron1 == "Late" and patron2 == "Intermediate":
        diag = """
### 🟦🟨 Sistema tardío pero distribuido
- Control pos-emergente es la herramienta principal.
- Riesgo de re-emergencias que superen la ventana óptima del graminicida.
"""
    else:
        diag = """
### Diagnóstico general
- El patrón dominante guía la estrategia de manejo.
- Los patrones secundarios ajustan la ventana óptima de control.
- Recomendado: combinar residual + post-emergente alrededor del JD50 estimado.
"""

    st.markdown(diag)
    st.success(
        f"Recomendación clave: priorizar el manejo según **{patron1}**, "
        "ajustando ventanas y persistencia de residuales según el patrón secundario."
    )


# ===============================================================
# DESCARGA DE SERIE ANN
# ===============================================================
st.download_button(
    "📥 Descargar EMERREL/EMERAC simulada (ANN)",
    df_ann.to_csv(index=False).encode("utf-8"),
    "emergencia_simulada_ANN_v85.csv",
    mime="text/csv"
)

