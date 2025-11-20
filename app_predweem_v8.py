# ===============================================================
# 🌾 PREDWEEM v8.3 — AVEFA Predictor 2026 (Con ANN + Clasificación)
# - ENTRENAMIENTO INTERNO meteo→patrón usando centroides
# - ANN → EMERREL diaria + EMERAC acumulada
# - Percentiles d25–d95 (curva ANN) + Radar año vs patrón
# - Certeza diaria del patrón (probabilidad día a día)
# - Comparación con curva observada + RMSE
# - Gráfico comparativo superpuesto + comparativo visual profesional
# - Comparación con patrón más cercano por centroides + año representativo
# - Siempre muestra los módulos; si falta info → mensajes internos
# - Compatible con meteo_daily.csv (Julian_days, TMAX, TMIN, Prec)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM v8.3 — AVEFA 2026", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 PREDWEEM v8.3 — AVEFA Predictor 2026")
st.subheader("Clasificación meteorológica + Emergencia simulada por ANN")

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 🔧 FUNCIONES AUXILIARES
# ===============================================================
def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

# ---------------------------------------------------------
# 📐 RMSE ENTRE DOS CURVAS
# ---------------------------------------------------------
def rmse_curvas(y_pred, y_obs):
    y_pred = np.asarray(y_pred, float)
    y_obs  = np.asarray(y_obs, float)
    return float(np.sqrt(np.mean((y_pred - y_obs)**2)))

# ---------------------------------------------------------
# 📈 GRÁFICO EMERGENCIA ACUMULADA (PREDICHA VS OBSERVADA)
# ---------------------------------------------------------
def plot_emergencia_acumulada(dias, emerac_pred, emerac_obs, nombre_obs="Observada"):
    fig, ax = plt.subplots(figsize=(9, 5))

    emerac_pred = np.asarray(emerac_pred, float)
    emerac_obs  = np.asarray(emerac_obs, float)

    if emerac_pred.max() > 0:
        y_pred = emerac_pred / emerac_pred.max()
    else:
        y_pred = emerac_pred

    if emerac_obs.max() > 0:
        y_obs = emerac_obs / emerac_obs.max()
    else:
        y_obs = emerac_obs

    ax.plot(dias, y_pred, label="Predicha (ANN)", color="blue", linewidth=3)
    ax.plot(dias, y_obs, label=nombre_obs, color="red", linestyle="--", linewidth=2)

    ax.set_xlabel("Día Juliano")
    ax.set_ylabel("Emergencia acumulada (0–1)")
    ax.set_title("Emergencia acumulada — Predicha vs Observada")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig

# ---------------------------------------------------------
# 📈 GRÁFICO COMPARATIVO SUPERPUESTO (CURVAS NORMALIZADAS)
# ---------------------------------------------------------
def plot_comparativo_curvas(jd, emerac_pred, emerac_obs, nombre_obs="Observada"):
    fig, ax = plt.subplots(figsize=(10, 5))

    emerac_pred = np.asarray(emerac_pred, float)
    emerac_obs  = np.asarray(emerac_obs, float)

    pred_norm = emerac_pred / emerac_pred.max() if emerac_pred.max() > 0 else emerac_pred
    obs_norm  = emerac_obs  / emerac_obs.max()  if emerac_obs.max()  > 0 else emerac_obs

    ax.plot(jd, pred_norm, color="blue", linewidth=3, label="Predicha (ANN)")
    ax.plot(jd, obs_norm,  color="red", linestyle="--", linewidth=2, label=nombre_obs)

    ax.set_xlabel("Día Juliano")
    ax.set_ylabel("Emergencia acumulada (normalizada)")
    ax.set_title("Comparación de curvas — EMERAC predicha vs observada")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return fig

# ---------------------------------------------------------
# 🎨 GRÁFICO COMPARATIVO VISUAL (ANN vs Observada)
# ---------------------------------------------------------
def plot_comparativo_visual(jd, emerac_pred, emerac_obs, 
                            perc_pred=None, perc_obs=None, nombre_obs="Observada"):
    fig, ax = plt.subplots(figsize=(12, 6))

    emerac_pred = np.asarray(emerac_pred, float)
    emerac_obs  = np.asarray(emerac_obs, float)

    pred = emerac_pred / emerac_pred.max() if emerac_pred.max() > 0 else emerac_pred
    obs  = emerac_obs  / emerac_obs.max()  if emerac_obs.max()  > 0 else emerac_obs

    ax.plot(jd, pred, color="blue", linewidth=3, label="Predicha (ANN)")
    ax.plot(jd, obs,  color="red", linestyle="--", linewidth=2, label=nombre_obs)

    ax.fill_between(jd, pred, obs, color="gray", alpha=0.25,
                    label="Diferencia |Pred - Obs|")

    if perc_pred is not None:
        for p, c in zip(perc_pred, ["#0033aa", "#0044dd", "#0055ff", "#0077ff"]):
            ax.axvline(p, color=c, linestyle="-", alpha=0.7, linewidth=1.8)

    if perc_obs is not None:
        for p, c in zip(perc_obs, ["#aa0000", "#cc0000", "#ee0000", "#ff2222"]):
            ax.axvline(p, color=c, linestyle="--", alpha=0.8, linewidth=1.7)

    ax.set_xlabel("Día Juliano")
    ax.set_ylabel("Emergencia acumulada normalizada (0–1)")
    ax.set_title("Comparación visual — EMERAC Predicha vs Observada")
    ax.grid(alpha=0.25)
    ax.legend()

    return fig

# ===============================================================
# 🔵 1. CURVAS DE EMERAC HISTÓRICAS 1977–2015 → JD25–95 + PATRÓN
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
    out = []
    for q in qs:
        out.append(float(np.interp(q, y, jd)))
    return np.array(out, dtype=float)

def _load_curves_emereac():
    curvas = {}

    xls1 = pd.ExcelFile(BASE / "emergencia_acumulada_interpolada 1977-1998.xlsx")
    for sh in xls1.sheet_names:
        df = pd.read_excel(xls1, sheet_name=sh)
        year = int(str(sh).split("_")[-1])
        curvas[year] = df[["JD", "EMERAC"]].copy()

    xls2 = pd.ExcelFile(BASE / "emergencia_2000_2015_interpolada.xlsx")
    for sh in xls2.sheet_names:
        df = pd.read_excel(xls2, sheet_name=sh)
        year = int(str(sh).split("_")[-1])
        curvas[year] = df[["JD", "EMERAC"]].copy()

    return curvas

def _assign_labels_from_centroids(curvas):
    cent = joblib.load(BASE / "predweem_model_centroides.pkl")
    C = cent["centroides"]   # index=patrones, cols=[JD25, JD50, JD75, JD95]

    registros = []
    for year, df in sorted(curvas.items()):
        vals = _compute_jd_percentiles(df["JD"], df["EMERAC"])
        if vals is None:
            continue
        d25, d50, d75, d95 = vals
        v = np.array([d25, d50, d75, d95])

        dists = ((C.values - v)**2).sum(axis=1)**0.5
        patron = C.index[np.argmin(dists)]

        registros.append({
            "anio": int(year),
            "patron": str(patron),
            "JD25": d25,
            "JD50": d50,
            "JD75": d75,
            "JD95": d95
        })

    return pd.DataFrame(registros)

@st.cache_resource
def load_hist_data_and_labels():
    curvas = _load_curves_emereac()
    labels_df = _assign_labels_from_centroids(curvas)
    return curvas, labels_df

# ===============================================================
# 🔵 2. FEATURES METEOROLÓGICOS HISTÓRICOS (Bordenave 1977–2015)
# ===============================================================
def _build_meteo_features_for_years(labels_df):
    xls = pd.ExcelFile(BASE / "Bordenave_1977_2015_por_anio_con_JD.xlsx")
    rows = []

    for _, row in labels_df.iterrows():
        year = int(row["anio"])
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
        df["TMIN"] = pd.to_numeric(df.get("TMIN"), errors="coerce")
        df["TMAX"] = pd.to_numeric(df.get("TMAX"), errors="coerce")
        df["Prec"] = pd.to_numeric(df.get("Prec"), errors="coerce")

        df = df.dropna(subset=["JD"])
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

# ===============================================================
# 🔵 3. FEATURES PARA ARCHIVOS METEOROLÓGICOS NUEVOS
# ===============================================================
def _build_features_from_df_meteo(df_meteo):
    df = df_meteo.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            n_low = n.lower()
            for col_low, col_orig in cols.items():
                if n_low in col_low:
                    return col_orig
        return None

    c_jd = None
    for col in df.columns:
        if col.lower() == "julian_days":
            c_jd = col
            break
    if c_jd is None:
        c_jd = pick("jd", "dia", "julian", "doy")

    c_tmin = pick("tmin", "temperatura_minima")
    c_tmax = pick("tmax", "temperatura_maxima")
    c_prec = pick("prec", "lluvia", "ppt", "prcp", "pluviometrica")
    c_fecha = pick("fecha", "date")

    if None in (c_jd, c_tmin, c_tmax, c_prec):
        raise ValueError("No se identificaron correctamente JD/TMIN/TMAX/Prec en el archivo cargado.")

    df["JD"]   = pd.to_numeric(df[c_jd], errors="coerce")
    df["TMIN"] = pd.to_numeric(df[c_tmin], errors="coerce")
    df["TMAX"] = pd.to_numeric(df[c_tmax], errors="coerce")
    df["Prec"] = pd.to_numeric(df[c_prec], errors="coerce")
    if c_fecha is not None:
        df["Fecha"] = pd.to_datetime(df[c_fecha], errors="coerce")

    df = df.dropna(subset=["JD"])
    df = df.sort_values("JD")

    df["Tmed"] = (df["TMIN"] + df["TMAX"]) / 2

    feats = {
        "Tmin_mean": df["TMIN"].mean(),
        "Tmax_mean": df["TMAX"].mean(),
        "Tmed_mean": df["Tmed"].mean(),
        "Prec_total": df["Prec"].sum(),
        "Prec_days_10mm": (df["Prec"] >= 10).sum(),
    }

    sub = df[df["JD"] <= 121]
    feats["Tmed_FM"] = sub["Tmed"].mean()
    feats["Prec_FM"] = sub["Prec"].sum()

    return pd.DataFrame([feats]), df

# ===============================================================
# 🔵 4. ENTRENAMIENTO INTERNO CLASIFICADOR METEO → PATRÓN
# ===============================================================
@st.cache_resource
def load_clf():
    curvas, labels_df = load_hist_data_and_labels()
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

def predecir_patron(df_meteo):
    model = load_clf()
    Xnew, df_limpio = _build_features_from_df_meteo(df_meteo)
    proba = model.predict_proba(Xnew)[0]
    clases = model.classes_
    pred = clases[np.argmax(proba)]
    return str(pred), dict(zip(map(str, clases), map(float, proba))), df_limpio

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
    IW  = np.load(BASE/"IW.npy")
    bIW = np.load(BASE/"bias_IW.npy")
    LW  = np.load(BASE/"LW.npy")
    bLW = np.load(BASE/"bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)

def postprocess_emergence(emerrel_raw, smooth=True, window=3, clip_zero=True):
    emer = np.array(emerrel_raw, dtype=float)
    if clip_zero:
        emer = np.maximum(emer, 0.0)
    if smooth and len(emer) > 1 and window > 1:
        window = int(window)
        window = max(1, min(window, len(emer)))
        kernel = np.ones(window)/window
        emer = np.convolve(emer, kernel, mode="same")
    emerac = np.cumsum(emer)
    return emer, emerac

# ===============================================================
# 🔵 6. RADAR MULTISERIES JD25–95 (Año ANN vs Patrón)
# ===============================================================
def radar_multiseries(values_dict, labels, title):
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    for name, vals in values_dict.items():
        vals2 = list(vals) + [vals[0]]
        ax.plot(angles, vals2, lw=2.5, label=name)
        ax.fill(angles, vals2, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.1))

    return fig

# ===============================================================
# 🔵 7. CONTROLES ANN EN SIDEBAR
# ===============================================================
with st.sidebar:
    st.header("Ajustes de emergencia (ANN)")
    use_smoothing = st.checkbox("Suavizar EMERREL", value=True)
    window_size   = st.slider("Ventana de suavizado (días)", 1, 9, 3)
    clip_zero     = st.checkbox("Recortar negativos a 0", value=True)

# ===============================================================
# 🔵 8. INTERFAZ PRINCIPAL — CARGA DE ARCHIVO
# ===============================================================
st.subheader("📤 Subir archivo meteorológico")
uploaded = st.file_uploader("Cargar archivo (ej. meteo_daily.csv):", type=["csv", "xlsx"])

modelo_ann = safe(load_ann, "Error cargando pesos ANN (IW.npy, bias_IW.npy, LW.npy, bias_out.npy)")

if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)

    st.success("Archivo cargado correctamente.")
    st.dataframe(df_raw, use_container_width=True)

    # -----------------------------------------------------------
    # CLASIFICACIÓN METEOROLÓGICA
    # -----------------------------------------------------------
    df_meteo = None
    patron_pred = None
    probs = None

    try:
        patron_pred, probs, df_meteo = predecir_patron(df_raw)
    except Exception as e:
        st.error(f"Error en la clasificación meteorológica: {e}")

    if patron_pred is not None and probs is not None:
        st.markdown(f"## 🌱 Patrón meteo→predicho: **{patron_pred}**")
        st.json(probs)
    else:
        st.info("No se pudo determinar el patrón meteorológico para este archivo.")

    # -----------------------------------------------------------
    # ANN — CÁLCULO DE EMERGENCIA
    # -----------------------------------------------------------
    st.subheader("🔍 Emergencia simulada por ANN (EMERREL / EMERAC)")

    df_ann = None
    dias = None
    emerrel = None
    emerac = None
    emerac_raw = None
    ann_ok = False

    if (modelo_ann is None) or (df_meteo is None):
        st.info("No se pudo ejecutar la ANN porque falta el modelo o el dataframe meteorológico limpio.")
    else:
        if not all(c in df_meteo.columns for c in ["JD", "TMAX", "TMIN", "Prec"]):
            st.info("No se identificaron correctamente las columnas JD/TMAX/TMIN/Prec para ejecutar la ANN.")
        else:
            try:
                df_ann = df_meteo.copy()
                df_ann = df_ann.sort_values("JD")
                dias = df_ann["JD"].to_numpy()

                X_ann = df_ann[["JD", "TMAX", "TMIN", "Prec"]].to_numpy(float)
                emerrel_raw, emerac_raw = modelo_ann.predict(X_ann)
                emerrel, emerac = postprocess_emergence(
                    emerrel_raw,
                    smooth=use_smoothing,
                    window=window_size,
                    clip_zero=clip_zero,
                )

                df_ann["EMERREL"] = emerrel
                df_ann["EMERAC"]  = emerac
                ann_ok = True

                col_er, col_ac = st.columns(2)

                with col_er:
                    fig_er, ax_er = plt.subplots(figsize=(5, 4))
                    ax_er.plot(dias, emerrel_raw, label="EMERREL cruda (ANN)", color="red", alpha=0.6)
                    ax_er.plot(dias, emerrel,     label="EMERREL procesada",   color="blue", linewidth=2)
                    ax_er.set_xlabel("Día juliano")
                    ax_er.set_ylabel("EMERREL (fracción diaria)")
                    ax_er.set_title("EMERREL: ANN vs post-proceso")
                    ax_er.legend()
                    st.pyplot(fig_er)

                with col_ac:
                    fig_ac, ax_ac = plt.subplots(figsize=(5, 4))
                    if emerac_raw[-1] > 0:
                        ax_ac.plot(dias, emerac_raw/emerac_raw[-1], label="EMERAC cruda (norm.)", color="orange", alpha=0.6)
                    else:
                        ax_ac.plot(dias, emerac_raw, label="EMERAC cruda", color="orange", alpha=0.6)
                    if emerac[-1] > 0:
                        ax_ac.plot(dias, emerac/emerac[-1], label="EMERAC procesada (norm.)", color="green", linewidth=2)
                    else:
                        ax_ac.plot(dias, emerac, label="EMERAC procesada", color="green", linewidth=2)
                    ax_ac.set_xlabel("Día juliano")
                    ax_ac.set_ylabel("EMERAC (0–1 relativo al período)")
                    ax_ac.set_title("EMERAC: ANN vs post-proceso")
                    ax_ac.legend()
                    st.pyplot(fig_ac)

            except Exception as e:
                st.error(f"Error calculando la emergencia con ANN: {e}")

    # ===============================================================
    # 🔵  COMPARACIÓN EMERGENCIA ACUMULADA — PREDICHA VS OBSERVADA
    # ===============================================================
    st.subheader("📊 Emergencia acumulada — Predicha vs Observada")

    emerac_obs_interp = None

    if not ann_ok:
        st.info("No hay curva ANN válida; la comparación con observada se realizará cuando se disponga de EMERAC predicha.")
    else:
        archivo_obs = st.file_uploader(
            "Cargar curva observada (JD + EMERAC o JD + EMERREL):",
            key="obs", type=["csv", "xlsx"]
        )

        if archivo_obs is not None:
            try:
                if archivo_obs.name.endswith(".csv"):
                    df_obs = pd.read_csv(archivo_obs)
                else:
                    df_obs = pd.read_excel(archivo_obs)

                col_jd = None
                for k in ["jd", "julian", "dia"]:
                    for c in df_obs.columns:
                        if k in c.lower():
                            col_jd = c
                            break
                    if col_jd:
                        break

                col_emerac = None
                col_emerrel = None
                for c in df_obs.columns:
                    if "emerac" in c.lower():
                        col_emerac = c
                    if "emerrel" in c.lower():
                        col_emerrel = c

                if col_jd is None or (col_emerac is None and col_emerrel is None):
                    st.error("No se detectaron columnas JD y EMERAC/EMERREL en la curva observada.")
                else:
                    jd_obs = pd.to_numeric(df_obs[col_jd], errors="coerce")
                    mask_val = ~jd_obs.isna()
                    jd_obs = jd_obs[mask_val]

                    if col_emerac is not None:
                        emerac_obs = pd.to_numeric(df_obs[col_emerac], errors="coerce")[mask_val]
                    else:
                        emerrel_obs = pd.to_numeric(df_obs[col_emerrel], errors="coerce")[mask_val]
                        emerac_obs = np.cumsum(np.maximum(emerrel_obs, 0))

                    jd_model = np.array(dias, float)
                    emerac_pred = np.array(emerac, float)

                    emerac_obs_interp = np.interp(jd_model, jd_obs, emerac_obs)

                    fig_cmp = plot_emergencia_acumulada(jd_model, emerac_pred, emerac_obs_interp)
                    st.pyplot(fig_cmp)

                    if emerac_pred.max() > 0:
                        y_pred_norm = emerac_pred / emerac_pred.max()
                    else:
                        y_pred_norm = emerac_pred

                    if emerac_obs_interp.max() > 0:
                        y_obs_norm = emerac_obs_interp / emerac_obs_interp.max()
                    else:
                        y_obs_norm = emerac_obs_interp

                    rmse_norm = rmse_curvas(y_pred_norm, y_obs_norm)
                    rmse_raw = rmse_curvas(emerac_pred, emerac_obs_interp)

                    st.markdown("### 📐 RMSE entre curvas")
                    st.write(f"- **RMSE normalizado (0–1):** `{rmse_norm:.5f}`")
                    st.write(f"- **RMSE crudo:** `{rmse_raw:.5f}`  (si ambas curvas están en escala comparable)")

                    
    # ===============================================================
    # 🔵 9. COMPARACIÓN CON PATRÓN MÁS CERCANO (CENTROIDES)
    # ===============================================================
    st.subheader("🌐 Comparación con patrón más cercano por centroides + año representativo")

    if (not ann_ok) or (patron_pred is None):
        st.info("No hay curva ANN válida o patrón meteo predicho; no se puede comparar con patrones históricos.")
    else:
        try:
            curvas_hist, labels_df = load_hist_data_and_labels()
            vals_ann = _compute_jd_percentiles(dias, emerac)

            if vals_ann is None:
                st.info("No se pudieron calcular percentiles d25–d95 de la curva ANN; no se puede comparar con patrones.")
            else:
                df_pat = labels_df[labels_df["patron"] == str(patron_pred)].copy()
                if df_pat.empty:
                    st.info(f"No hay años históricos etiquetados con el patrón {patron_pred}.")
                else:
                    V_ann = np.array(vals_ann)
                    mat_hist = df_pat[["JD25", "JD50", "JD75", "JD95"]].to_numpy(float)
                    dists = np.linalg.norm(mat_hist - V_ann, axis=1)
                    idx_min = int(np.argmin(dists))
                    fila_rep = df_pat.iloc[idx_min]
                    anio_rep = int(fila_rep["anio"])
                    dist_rep = float(dists[idx_min])

                    st.write(f"- **Patrón meteo predicho:** `{patron_pred}`")
                    st.write(f"- **Año histórico representativo (centroides):** `{anio_rep}`")
                    st.write(f"- **Distancia en espacio de percentiles (JD25–95):** `{dist_rep:.2f}`")

                    if anio_rep not in curvas_hist:
                        st.info(f"No se encontró la curva EMERAC histórica para el año {anio_rep}.")
                    else:
                        df_rep = curvas_hist[anio_rep].copy()
                        jd_rep = pd.to_numeric(df_rep["JD"], errors="coerce")
                        emerac_rep = pd.to_numeric(df_rep["EMERAC"], errors="coerce")

                        mask = ~jd_rep.isna() & ~emerac_rep.isna()
                        jd_rep = jd_rep[mask].to_numpy()
                        emerac_rep = emerac_rep[mask].to_numpy()

                        emerac_rep_interp = np.interp(dias, jd_rep, emerac_rep)

                        fig_pat = plot_emergencia_acumulada(
                            dias,
                            emerac,
                            emerac_rep_interp,
                            nombre_obs=f"Año hist. {anio_rep} (patrón {patron_pred})"
                        )
                        st.pyplot(fig_pat)

                        st.markdown("### 📐 RMSE ANN vs año representativo (EMERAC)")

                        if emerac.max() > 0:
                            pred_norm_rep = emerac / emerac.max()
                        else:
                            pred_norm_rep = emerac
                        if emerac_rep_interp.max() > 0:
                            rep_norm = emerac_rep_interp / emerac_rep_interp.max()
                        else:
                            rep_norm = emerac_rep_interp

                        rmse_norm_rep = rmse_curvas(pred_norm_rep, rep_norm)
                        rmse_raw_rep  = rmse_curvas(emerac, emerac_rep_interp)

                        st.write(f"- **RMSE normalizado (0–1):** `{rmse_norm_rep:.5f}`")
                        st.write(f"- **RMSE crudo:** `{rmse_raw_rep:.5f}`")

                        st.subheader("📈 Curvas comparativas — ANN vs Año representativo")
                        fig_super_rep = plot_comparativo_curvas(
                            dias,
                            emerac,
                            emerac_rep_interp,
                            nombre_obs=f"Año {anio_rep} (patrón {patron_pred})"
                        )
                        st.pyplot(fig_super_rep)

        except Exception as e:
            st.error(f"No se pudo comparar con el patrón más cercano: {e}")

    # ===============================================================
    # 🔵 10. Percentiles ANN del año (sobre lo emergido)
    # ===============================================================
    st.subheader("📌 Percentiles ANN del año (sobre lo emergido)")

    if not ann_ok:
        st.info("No hay curva ANN válida para calcular percentiles.")
    else:
        vals = _compute_jd_percentiles(dias, emerac)
        if vals is not None:
            d25, d50, d75, d95 = vals
            st.write({
                "d25": round(d25, 1),
                "d50": round(d50, 1),
                "d75": round(d75, 1),
                "d95": round(d95, 1),
            })
        else:
            st.info("No se pudieron calcular percentiles d25–d95 para esta curva ANN.")

    # ===============================================================
    # 🔵 11. Radar JD25–95: Año ANN vs patrón meteo
    # ===============================================================
    st.subheader("🎯 Radar JD25–95: Año ANN vs patrón meteo")

    if (not ann_ok) or (patron_pred is None):
        st.info("Se necesita una curva ANN válida y un patrón meteo predicho para generar el radar.")
    else:
        vals = _compute_jd_percentiles(dias, emerac)
        if vals is None:
            st.info("No se pudieron calcular percentiles ANN para el radar.")
        else:
            d25, d50, d75, d95 = vals
            try:
                cent = joblib.load(BASE / "predweem_model_centroides.pkl")
                C = cent["centroides"]
                if patron_pred in C.index:
                    vals_year = [d25, d50, d75, d95]
                    vals_pat  = list(C.loc[patron_pred][["JD25","JD50","JD75","JD95"]].values)
                    fig_rad = radar_multiseries(
                        {
                            "Año evaluado (ANN)": vals_year,
                            f"Patrón {patron_pred}": vals_pat
                        },
                        labels=["d25","d50","d75","d95"],
                        title="Radar — Año vs patrón"
                    )
                    st.pyplot(fig_rad)
                else:
                    st.info("El patrón predicho no se encuentra en los centroides.")
            except Exception as e:
                st.error(f"No se pudo generar el radar comparativo: {e}")

    # ===============================================================
    # 🔵 12. Certeza diaria del patrón (ANN + centroides)
    # ===============================================================
    st.subheader("📈 Certeza diaria del patrón (ANN + centroides)")

    if (not ann_ok) or (patron_pred is None):
        st.info("Se requiere una curva ANN válida y un patrón predicho para calcular la certeza diaria.")
    else:
        try:
            cent = joblib.load(BASE / "predweem_model_centroides.pkl")
            C = cent["centroides"]

            if patron_pred not in C.index:
                st.warning(f"El patrón predicho (**{patron_pred}**) no se encuentra en los centroides.")
            else:
                jd_eval = []
                fechas_eval = []
                probs_sel = []

                fecha_col = None
                for c in df_ann.columns:
                    if "fecha" in c.lower() or "date" in c.lower():
                        fecha_col = c
                        break
                if fecha_col is not None:
                    df_ann[fecha_col] = pd.to_datetime(df_ann[fecha_col], errors="coerce")

                for i in range(4, len(dias)):
                    jd_sub = dias[:i+1]
                    emerac_sub = emerac[:i+1]

                    vals_i = _compute_jd_percentiles(jd_sub, emerac_sub)
                    if vals_i is None:
                        continue

                    v = np.array(vals_i)
                    dists = np.linalg.norm(C.values - v, axis=1)
                    w = 1.0 / (dists + 1e-6)
                    p = w / w.sum()

                    idx_pat = list(C.index).index(patron_pred)
                    p_sel = float(p[idx_pat])

                    jd_eval.append(jd_sub[-1])
                    probs_sel.append(p_sel)

                    if fecha_col is not None and fecha_col in df_ann.columns:
                        subf = df_ann[df_ann["JD"] == jd_sub[-1]]
                        if not subf.empty:
                            fechas_eval.append(subf[fecha_col].max())
                        else:
                            fechas_eval.append(None)
                    else:
                        fechas_eval.append(None)

                if len(jd_eval) == 0:
                    st.info("No se pudo calcular la evolución diaria del patrón (curva ANN demasiado corta).")
                else:
                    TEMPORADA_MAX = 274
                    JD_START = int(dias.min())
                    JD_END   = int(dias.max())
                    cobertura = (JD_END - JD_START + 1) / TEMPORADA_MAX

                    UMBRAL = 0.8
                    idx_crit = next((i for i, p in enumerate(probs_sel) if p >= UMBRAL), None)
                    idx_max  = int(np.argmax(probs_sel)) if len(probs_sel) > 0 else None

                    fecha_crit = None
                    prob_crit  = None
                    if idx_crit is not None:
                        prob_crit = probs_sel[idx_crit]
                        fecha_crit = fechas_eval[idx_crit]

                    fecha_max = None
                    prob_max  = None
                    if idx_max is not None:
                        prob_max = probs_sel[idx_max]
                        fecha_max = fechas_eval[idx_max]

                    figp, axp = plt.subplots(figsize=(9, 5))

                    if fecha_col is not None and any(f is not None for f in fechas_eval):
                        x_axis = [f if f is not None else pd.NaT for f in fechas_eval]
                        axp.set_xlabel("Fecha calendario")
                    else:
                        x_axis = jd_eval
                        axp.set_xlabel("Día juliano")

                    axp.plot(
                        x_axis, probs_sel,
                        label=f"P({patron_pred}) según ANN+centroides",
                        color="green", lw=2
                    )

                    if idx_crit is not None:
                        axp.axvline(
                            x_axis[idx_crit],
                            color="green", linestyle="--", linewidth=2,
                            label=f"Momento crítico (P ≥ {UMBRAL:.0%})"
                        )
                    if idx_max is not None and (idx_crit is None or idx_max != idx_crit):
                        axp.axvline(
                            x_axis[idx_max],
                            color="blue", linestyle=":", linewidth=2,
                            label="Máxima certeza"
                        )

                    axp.set_ylim(0, 1)
                    axp.set_ylabel("Probabilidad del patrón seleccionado")
                    axp.set_title("Evolución diaria de la certeza del patrón")
                    axp.legend()
                    figp.autofmt_xdate()
                    st.pyplot(figp)

                    st.markdown("### 🧠 Momento crítico de definición del patrón (ANN + centroides)")

                    def fmt_fecha(idx):
                        f = fechas_eval[idx]
                        if isinstance(f, pd.Timestamp):
                            return f.strftime("%d-%b")
                        else:
                            return f"JD {jd_eval[idx]}"

                    if idx_crit is not None and prob_crit is not None:
                        st.write(
                            f"- **Patrón resultante:** {patron_pred}  \n"
                            f"- **Momento crítico (primer día con P≥{UMBRAL:.0%}):** "
                            f"**{fmt_fecha(idx_crit)}** (P = {prob_crit:.2f})  \n"
                            f"- **Máxima certeza:** **{fmt_fecha(idx_max)}** "
                            f"(P = {prob_max:.2f})"
                        )
                    elif idx_max is not None and prob_max is not None:
                        st.write(
                            f"- **Patrón resultante:** {patron_pred}  \n"
                            f"- No se alcanzó el umbral de {UMBRAL:.0%}, "
                            f"pero la máxima certeza se logra el **{fmt_fecha(idx_max)}** "
                            f"con P = **{prob_max:.2f}**."
                        )
                    else:
                        st.info("No se pudo calcular un resumen de certeza temporal.")

                    if prob_max is not None:
                        if cobertura >= 0.7 and prob_max >= 0.8:
                            nivel_conf = "ALTA"
                            color_conf = "green"
                        elif cobertura >= 0.4 and prob_max >= 0.65:
                            nivel_conf = "MEDIA"
                            color_conf = "orange"
                        else:
                            nivel_conf = "BAJA"
                            color_conf = "red"

                        st.markdown(
                            f"### 🔒 Nivel de confianza global (ANN + centroides): "
                            f"<span style='color:{color_conf}; font-size:26px;'>{nivel_conf}</span>",
                            unsafe_allow_html=True
                        )
                        st.write(
                            f"- **Cobertura temporal:** {cobertura*100:.1f} % de la temporada (1-ene→1-oct)  \n"
                            f"- **Probabilidad máxima del patrón seleccionado:** {prob_max:.2f}"
                        )
        except Exception as e:
            st.error(f"No se pudo calcular la certeza diaria del patrón (ANN+centroides): {e}")

    # ===============================================================
    # 🔵 13. Descarga de serie ANN
    # ===============================================================
    if ann_ok and df_ann is not None:
        csv_ann = df_ann.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar EMERREL/EMERAC simulada",
            csv_ann,
            "emergencia_simulada_ANN.csv",
            mime="text/csv"
        )
else:
    st.info("Cargue un archivo meteorológico para iniciar el análisis.")

