# ===============================================================
# 🌾 PREDWEEM v8.3 — AVEFA Predictor 2026 (ANN + Centroides)
# - Patrón definido por curva ANN comparada con centroides
# - SIN gráfico simple "Predicha vs Observada"
# - Incluye siempre (con mensajes internos):
#   * EMERREL / EMERAC (ANN)
#   * Clasificación de patrón (ANN+centroides)
#   * Comparación con curva observada (RMSE + gráficos)
#   * Comparación con patrón más cercano + año representativo
#   * Radar JD25–95
#   * Certeza diaria del patrón
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
st.subheader("Curva ANN + Centroides → Patrón + Comparaciones")

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 🔧 FUNCIONES AUXILIARES GENERALES
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
# 📈 GRÁFICOS COMPARATIVOS ANN vs OBSERVADA
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
    ax.set_ylabel("Emergencia acumulada normalizada (0–1)")
    ax.set_title("Comparación superpuesta — ANN vs Observada")
    ax.grid(alpha=0.25)
    ax.legend()

    return fig

def plot_comparativo_visual(jd, emerac_pred, emerac_obs,
                            perc_pred=None, perc_obs=None,
                            nombre_obs="Observada"):
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
    ax.set_title("Comparativo visual — ANN vs Observada")
    ax.grid(alpha=0.25)
    ax.legend()

    return fig

# ===============================================================
# 🔵 PERCENTILES + HISTÓRICO + CENTROIDES
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
    """
    Lee curvas EMERAC históricas desde:
    - emergencia_acumulada_interpolada 1977-1998.xlsx
    - emergencia_2000_2015_interpolada.xlsx
    Devuelve dict: year -> DataFrame[JD, EMERAC]
    """
    curvas = {}

    # 1977–1998
    try:
        xls1 = pd.ExcelFile(BASE / "emergencia_acumulada_interpolada 1977-1998.xlsx")
        for sh in xls1.sheet_names:
            df = pd.read_excel(xls1, sheet_name=sh)
            year = int(str(sh).split("_")[-1])
            curvas[year] = df[["JD", "EMERAC"]].copy()
    except Exception:
        pass

    # 2000–2015
    try:
        xls2 = pd.ExcelFile(BASE / "emergencia_2000_2015_interpolada.xlsx")
        for sh in xls2.sheet_names:
            df = pd.read_excel(xls2, sheet_name=sh)
            year = int(str(sh).split("_")[-1])
            curvas[year] = df[["JD", "EMERAC"]].copy()
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
        v = np.array([d25, d50, d75, d95])

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
    """
    Carga centroides, histórico y año representativo por patrón.
    """
    cent = joblib.load(BASE / "predweem_model_centroides.pkl")
    C = cent["centroides"]
    curvas = _load_curves_emereac()
    labels_df = _assign_labels_from_centroids(curvas, C)

    rep_year = {}
    for patron in C.index:
        sub = labels_df[labels_df["patron"] == str(patron)]
        if sub.empty:
            continue
        vc = C.loc[patron][["JD25", "JD50", "JD75", "JD95"]].values.astype(float)
        M = sub[["JD25", "JD50", "JD75", "JD95"]].values.astype(float)
        dists = np.linalg.norm(M - vc, axis=1)
        best = int(np.argmin(dists))
        rep_year[str(patron)] = int(sub.iloc[best]["anio"])

    return C, labels_df, rep_year, curvas

def clasificar_patron_desde_ann(dias, emerac, C):
    """
    Usa solo la curva ANN (EMERAC) comparada con centroides para definir el patrón.
    """
    vals = _compute_jd_percentiles(dias, emerac)
    if vals is None:
        return None, None, None, None

    v = np.array(vals, float)
    dists = np.linalg.norm(C.values - v, axis=1)
    idx_min = int(np.argmin(dists))
    patron = str(C.index[idx_min])

    w = 1.0 / (dists + 1e-6)
    p = w / w.sum()
    prob_dict = {str(C.index[i]): float(p[i]) for i in range(len(C.index))}

    return patron, vals, dists, prob_dict

# ===============================================================
# 🔵 ANN — PREDICCIÓN DE EMERGENCIA
# ===============================================================
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
        # Rango de normalización de entradas [JD, TMAX, TMIN, Prec]
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
    emer = np.array(emerrel_raw, dtype=float)
    if clip_zero:
        emer = np.maximum(emer, 0.0)
    if smooth and len(emer) > 1 and window > 1:
        window = int(window)
        window = max(1, min(window, len(emer)))
        kernel = np.ones(window) / window
        emer = np.convolve(emer, kernel, mode="same")
    emerac = np.cumsum(emer)
    return emer, emerac

# ===============================================================
# 🔵 RADAR MULTISERIES JD25–95
# ===============================================================
def radar_multiseries(values_dict, labels, title):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
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
# 🔵 NORMALIZAR ARCHIVO METEOROLÓGICO
# ===============================================================
def normalizar_meteo(df_meteo):
    """
    Detecta columnas JD / TMAX / TMIN / Prec / Fecha
    y devuelve DataFrame con esas columnas estandarizadas.
    """
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

    return df


# ===============================================================
# 🔵 CONTROLES ANN EN SIDEBAR
# ===============================================================
with st.sidebar:
    st.header("Ajustes de emergencia (ANN)")
    use_smoothing = st.checkbox("Suavizar EMERREL", value=True)
    window_size   = st.slider("Ventana de suavizado (días)", 1, 9, 3)
    clip_zero     = st.checkbox("Recortar negativos a 0", value=True)

# ===============================================================
# 🔵 INTERFAZ PRINCIPAL — CARGA DE ARCHIVO METEOROLÓGICO
# ===============================================================
st.subheader("📤 Subir archivo meteorológico")
uploaded = st.file_uploader("Cargar archivo (ej. meteo_daily.csv):", type=["csv", "xlsx"])

modelo_ann = safe(load_ann, "Error cargando pesos ANN (IW.npy, bias_IW.npy, LW.npy, bias_out.npy)")

if uploaded is None:
    st.info("Cargue un archivo meteorológico para iniciar el análisis (JD, TMAX, TMIN, Prec...).")
else:
    # Leer archivo original
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)

    st.success("Archivo cargado correctamente.")
    st.dataframe(df_raw, use_container_width=True)

    # Normalizar columnas meteo
    try:
        df_meteo = normalizar_meteo(df_raw)
    except Exception as e:
        st.error(f"No se pudo normalizar el archivo meteorológico: {e}")
        df_meteo = None

    if modelo_ann is None:
        st.warning("No se pudo cargar la ANN. Revise los archivos IW.npy, bias_IW.npy, LW.npy, bias_out.npy.")
        df_meteo = None  # para evitar seguir
    # ===========================================================
    # ANN solo si tenemos modelo y meteo normalizada
    # ===========================================================
    if df_meteo is not None and modelo_ann is not None:
        st.subheader("🔍 Emergencia simulada por ANN (EMERREL / EMERAC)")

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


        # ===============================================================
        # 🔵 CLASIFICACIÓN DE PATRÓN SEGÚN CURVA ANN + CENTROIDES
        # ===============================================================
        st.subheader("🌱 Patrón definido por curva ANN + centroides")

        C, labels_df, rep_year, curvas_hist = safe(
            lambda: load_centroides_y_historia(),
            "No se pudieron cargar centroides e histórico (predweem_model_centroides.pkl / archivos de EMERAC histórica)."
        ) or (None, None, None, None)

        patron_ann = None
        vals_ann   = None
        dists_ann  = None
        prob_dict  = None

        if C is None:
            st.info("Sin centroides no es posible definir el patrón a partir de la curva ANN.")
        else:
            patron_ann, vals_ann, dists_ann, prob_dict = clasificar_patron_desde_ann(dias, emerac, C)
            if patron_ann is None:
                st.info("La curva ANN no tiene suficiente emergencia para calcular percentiles JD25–95.")
            else:
                st.markdown(f"### ✅ Patrón ANN seleccionado: **{patron_ann}** (curva ANN vs centroides)")

                st.write("**Probabilidades relativas por patrón (ANN + centroides):**")
                st.json(prob_dict)

                # Distancias como tabla
                dist_table = pd.DataFrame({
                    "patron": [str(p) for p in C.index],
                    "distancia": list(dists_ann),
                    "prob_relativa": [prob_dict[str(p)] for p in C.index]
                }).sort_values("distancia")
                st.dataframe(dist_table, use_container_width=True)

                # Año representativo del patrón ANN
                if rep_year is not None and str(patron_ann) in rep_year:
                    ano_rep = rep_year[str(patron_ann)]
                    st.markdown(
                        f"- **Año histórico representativo** del patrón **{patron_ann}**: "
                        f"**{ano_rep}** (curva más cercana a su centroide)."
                    )
                else:
                    st.info("No se pudo identificar un año representativo para este patrón.")

        # ===============================================================
        # 📊 COMPARACIÓN CON CURVA OBSERVADA (RMSE + GRÁFICOS)
        # ===============================================================
        st.subheader("📊 Comparación con curva observada (RMSE + gráficos)")

        archivo_obs = st.file_uploader(
            "Cargar curva observada (JD + EMERAC o JD + EMERREL):",
            key="obs", type=["csv", "xlsx"]
        )

        emerac_obs_interp = None

        if archivo_obs is None:
            st.info("Suba una curva observada para calcular RMSE y ver los gráficos comparativos.")
        else:
            # Leer curva observada
            if archivo_obs.name.endswith(".csv"):
                df_obs = pd.read_csv(archivo_obs)
            else:
                df_obs = pd.read_excel(archivo_obs)

            # Detectar JD
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

                # Interpolar observada al eje del modelo
                emerac_obs_interp = np.interp(jd_model, jd_obs, emerac_obs)

                # RMSE entre curvas
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

                st.markdown("### 📐 RMSE entre curvas (ANN vs observada)")
                st.write(f"- **RMSE normalizado (0–1):** `{rmse_norm:.5f}`")
                st.write(f"- **RMSE crudo:** `{rmse_raw:.5f}`  (si ambas curvas están en escala comparable)")

                # Gráfico comparativo superpuesto
                st.subheader("📈 Curvas comparativas — ANN vs Observada")
                fig_super = plot_comparativo_curvas(
                    jd_model,
                    emerac_pred,         # EMERAC ANN procesada
                    emerac_obs_interp,   # EMERAC observada interpolada
                    nombre_obs="Curva observada"
                )
                st.pyplot(fig_super)

                # Comparativo visual con banda de error + percentiles
                perc_pred = _compute_jd_percentiles(jd_model, emerac_pred)
                perc_obs  = _compute_jd_percentiles(jd_model, emerac_obs_interp)

                if perc_pred is not None and perc_obs is not None:
                    st.subheader("🎨 Comparativo visual ANN vs Observada")
                    fig_visual = plot_comparativo_visual(
                        jd_model,
                        emerac_pred,
                        emerac_obs_interp,
                        perc_pred=perc_pred,
                        perc_obs=perc_obs,
                        nombre_obs="Curva observada"
                    )
                    st.pyplot(fig_visual)
                else:
                    st.info("No se pudieron calcular percentiles JD25–95 para alguna de las curvas.")


        # ===============================================================
        # 📌 Percentiles ANN del año (sobre lo emergido)
        # ===============================================================
        st.subheader("📌 Percentiles JD25–95 de la curva ANN")

        vals_ANN = _compute_jd_percentiles(dias, emerac)
        if vals_ANN is not None:
            d25, d50, d75, d95 = vals_ANN
            st.write({
                "d25": round(d25, 1),
                "d50": round(d50, 1),
                "d75": round(d75, 1),
                "d95": round(d95, 1),
            })
        else:
            st.info("La curva ANN no alcanza niveles suficientes de emergencia para calcular d25–d95.")

        # ===============================================================
        # 🎯 Radar JD25–95: Año ANN vs patrón ANN seleccionado
        # ===============================================================
        st.subheader("🎯 Radar JD25–95: Año ANN vs patrón por centroides")

        if patron_ann is None or C is None or vals_ann is None:
            st.info("No se puede generar el radar porque falta patrón ANN o centroides.")
        else:
            try:
                vals_pat = list(C.loc[patron_ann][["JD25", "JD50", "JD75", "JD95"]].values)
                fig_rad = radar_multiseries(
                    {
                        "Año evaluado (ANN)": list(vals_ann),
                        f"Patrón {patron_ann}": vals_pat,
                    },
                    labels=["d25", "d50", "d75", "d95"],
                    title="Radar — Año ANN vs Patrón ANN"
                )
                st.pyplot(fig_rad)
            except Exception as e:
                st.error(f"No se pudo generar el radar comparativo: {e}")

        # ===============================================================
        # 📊 Comparación con curva del año representativo del patrón
        # ===============================================================
        st.subheader("📊 Comparación con año histórico representativo del patrón ANN")

        if patron_ann is None or C is None or rep_year is None or curvas_hist is None:
            st.info("No se puede comparar con año representativo (faltan centroides o histórico).")
        else:
            if str(patron_ann) not in rep_year:
                st.info("No hay año representativo disponible para este patrón.")
            else:
                yr_rep = rep_year[str(patron_ann)]
                if yr_rep not in curvas_hist:
                    st.info(f"No se encontró la curva histórica de EMERAC para el año {yr_rep}.")
                else:
                    df_rep = curvas_hist[yr_rep].copy()
                    jd_rep = df_rep["JD"].to_numpy(float)
                    em_rep = df_rep["EMERAC"].to_numpy(float)

                    # Interpolar la curva representativa al eje del modelo
                    jd_model = np.array(dias, float)
                    em_rep_interp = np.interp(jd_model, jd_rep, em_rep)

                    fig_rep = plot_comparativo_curvas(
                        jd_model,
                        emerac,          # curva ANN
                        em_rep_interp,   # curva año representativo
                        nombre_obs=f"Año representativo {yr_rep}"
                    )
                    st.pyplot(fig_rep)

        # ===============================================================
        # 📈 Certeza diaria del patrón (ANN + centroides)
        # ===============================================================
        st.subheader("📈 Certeza diaria del patrón (ANN + centroides)")

        if patron_ann is None or C is None:
            st.info("No se puede calcular la certeza diaria sin patrón ANN y centroides.")
        else:
            try:
                jd_eval = []
                probs_sel = []
                fechas_eval = []

                # Detectar posible columna de fecha
                fecha_col = None
                for c in df_ann.columns:
                    if "fecha" in c.lower() or "date" in c.lower():
                        fecha_col = c
                        break
                if fecha_col is not None:
                    df_ann[fecha_col] = pd.to_datetime(df_ann[fecha_col], errors="coerce")

                for i in range(4, len(dias)):  # arrancar un poco más adelante
                    jd_sub = dias[:i+1]
                    emerac_sub = emerac[:i+1]

                    vals_i = _compute_jd_percentiles(jd_sub, emerac_sub)
                    if vals_i is None:
                        continue

                    v = np.array(vals_i)
                    dists = np.linalg.norm(C.values - v, axis=1)
                    w = 1.0 / (dists + 1e-6)
                    p = w / w.sum()

                    idx_pat = list(C.index).index(patron_ann)
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
                        label=f"P({patron_ann}) según ANN+centroides",
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
                            f"- **Patrón resultante:** {patron_ann}  \n"
                            f"- **Momento crítico (primer día con P≥{UMBRAL:.0%}):** "
                            f"**{fmt_fecha(idx_crit)}** (P = {prob_crit:.2f})  \n"
                            f"- **Máxima certeza:** **{fmt_fecha(idx_max)}** "
                            f"(P = {prob_max:.2f})"
                        )
                    elif idx_max is not None and prob_max is not None:
                        st.write(
                            f"- **Patrón resultante:** {patron_ann}  \n"
                            f"- No se alcanzó el umbral de {UMBRAL:.0%}, "
                            f"pero la máxima certeza se logra el **{fmt_fecha(idx_max)}** "
                            f"con P = **{prob_max:.2f}**."
                        )
                    else:
                        st.info("No se pudo calcular un resumen de certeza temporal.")

                    # Nivel de confianza global
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
        # 📥 Descarga de serie ANN
        # ===============================================================
        csv_ann = df_ann.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar EMERREL/EMERAC simulada",
            csv_ann,
            "emergencia_simulada_ANN.csv",
            mime="text/csv"
        )
















