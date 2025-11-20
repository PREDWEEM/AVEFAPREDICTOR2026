
# ===============================================================
# 🌾 PREDWEEM v8.3 — AVEFA Predictor 2026 (Con ANN + Clasificación)
# - Módulos SIEMPRE visibles con mensajes internos
# - SIN gráfico “Emergencia acumulada — Predicha vs Observada”
# - ANN + Certeza diaria + Radar + RMSE + Comparaciones visuales
# - Comparación con patrón más cercano + año representativo
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
st.subheader("Clasificación meteorológica + ANN + Comparaciones completas")

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 🔧 UTILS
# ===============================================================
def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

# ===============================================================
# 📐 RMSE ENTRE DOS CURVAS
# ===============================================================
def rmse_curvas(y_pred, y_obs):
    y_pred = np.asarray(y_pred, float)
    y_obs  = np.asarray(y_obs, float)
    return float(np.sqrt(np.mean((y_pred - y_obs)**2)))

# ===============================================================
# 📈 GRÁFICO SUPERPUESTO (Predicha vs Observada)
# ===============================================================
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

# ===============================================================
# 🎨 GRÁFICO VISUAL PROFESIONAL
# ===============================================================
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
# 🔵 Funciones de percentiles, carga histórica, centroides...
# ===============================================================
def _compute_jd_percentiles(jd, emerac, qs=(0.25,0.5,0.75,0.95)):
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
    return np.array(out, float)

def _load_curves_emereac():
    curvas = {}
    try:
        xls1 = pd.ExcelFile(BASE / "emergencia_acumulada_interpolada 1977-1998.xlsx")
        for sh in xls1.sheet_names:
            df = pd.read_excel(xls1, sheet_name=sh)
            year = int(str(sh).split("_")[-1])
            curvas[year] = df[["JD","EMERAC"]].copy()
    except Exception:
        pass

    try:
        xls2 = pd.ExcelFile(BASE / "emergencia_2000_2015_interpolada.xlsx")
        for sh in xls2.sheet_names:
            df = pd.read_excel(xls2, sheet_name=sh)
            year = int(str(sh).split("_")[-1])
            curvas[year] = df[["JD","EMERAC"]].copy()
    except Exception:
        pass

    return curvas

def _assign_labels_from_centroids(curvas):
    cent = joblib.load(BASE/"predweem_model_centroides.pkl")
    C = cent["centroides"]

    registros = []
    for year, df in curvas.items():
        vals = _compute_jd_percentiles(df["JD"], df["EMERAC"])
        if vals is None:
            continue
        d25, d50, d75, d95 = vals
        v = np.array([d25,d50,d75,d95])

        dists = np.linalg.norm(C.values - v, axis=1)
        patron = C.index[np.argmin(dists)]

        registros.append({
            "anio":year,
            "patron":str(patron),
            "JD25":d25, "JD50":d50, "JD75":d75, "JD95":d95
        })

    return pd.DataFrame(registros)

# ===============================================================
# 🔵 2. FEATURES METEOROLÓGICOS HISTÓRICOS
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
            "Temperatura_Minima":"TMIN",
            "Temperatura_Maxima":"TMAX",
            "Precipitacion_Pluviometrica":"Prec",
        }, inplace=True)

        df["JD"]   = pd.to_numeric(df["JD"], errors="coerce")
        df["TMIN"] = pd.to_numeric(df.get("TMIN"), errors="coerce")
        df["TMAX"] = pd.to_numeric(df.get("TMAX"), errors="coerce")
        df["Prec"] = pd.to_numeric(df.get("Prec"), errors="coerce")

        df = df.dropna(subset=["JD"])
        df["Tmed"] = (df["TMIN"] + df["TMAX"]) / 2

        feats = {
            "anio":year,
            "patron":patron,
            "Tmin_mean":df["TMIN"].mean(),
            "Tmax_mean":df["TMAX"].mean(),
            "Tmed_mean":df["Tmed"].mean(),
            "Prec_total":df["Prec"].sum(),
            "Prec_days_10mm":(df["Prec"] >= 10).sum(),
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
        c_jd = pick("jd","dia","julian","doy")

    c_tmin  = pick("tmin","temperatura_minima")
    c_tmax  = pick("tmax","temperatura_maxima")
    c_prec  = pick("prec","lluvia","ppt","prcp","pluviometrica")
    c_fecha = pick("fecha","date")

    if None in (c_jd, c_tmin, c_tmax, c_prec):
        raise ValueError("No se identificaron correctamente JD/TMIN/TMAX/Prec en el archivo cargado.")

    df["JD"]   = pd.to_numeric(df[c_jd],   errors="coerce")
    df["TMIN"] = pd.to_numeric(df[c_tmin], errors="coerce")
    df["TMAX"] = pd.to_numeric(df[c_tmax], errors="coerce")
    df["Prec"] = pd.to_numeric(df[c_prec], errors="coerce")

    if c_fecha is not None:
        df["Fecha"] = pd.to_datetime(df[c_fecha], errors="coerce")

    df = df.dropna(subset=["JD"])
    df = df.sort_values("JD")
    df["Tmed"] = (df["TMIN"] + df["TMAX"]) / 2

    feats = {
        "Tmin_mean":df["TMIN"].mean(),
        "Tmax_mean":df["TMAX"].mean(),
        "Tmed_mean":df["Tmed"].mean(),
        "Prec_total":df["Prec"].sum(),
        "Prec_days_10mm":(df["Prec"] >= 10).sum(),
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
    curvas = _load_curves_emereac()
    if not curvas:
        raise RuntimeError("No se cargaron curvas históricas de EMERAC.")
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
        self.input_min = np.array([1, 0, -7, 0])      # [JD, TMAX, TMIN, Prec]
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2*(X - self.input_min)/(self.input_max - self.input_min) - 1

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
    emer = np.array(emerrel_raw, float)
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
# 🔵 8. INTERFAZ PRINCIPAL — CARGA DE ARCHIVO METEOROLÓGICO
# ===============================================================
st.subheader("📤 Subir archivo meteorológico")
uploaded = st.file_uploader("Cargar archivo (ej. meteo_daily.csv):", type=["csv", "xlsx"])

modelo_ann = safe(load_ann, "Error cargando pesos ANN (IW.npy, bias_IW.npy, LW.npy, bias_out.npy)")

ann_ok = False
dias_ann = None
emerac_ann = None
vals_ann = None  # percentiles ANN

if uploaded is not None:
    # Leer archivo
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)

    st.success("Archivo cargado correctamente.")
    st.dataframe(df_raw, use_container_width=True)

    # --- CLASIFICACIÓN METEOROLÓGICA + DF LIMPIO PARA ANN ---
    try:
        patron_pred, probs, df_meteo = predecir_patron(df_raw)
        st.markdown(f"## 🌱 Patrón meteo→predicho: **{patron_pred}**")
        st.json(probs)
    except Exception as e:
        st.error(f"Error en la clasificación meteorológica: {e}")
        df_meteo = None

    # ===========================================================
    # 🔹 MÓDULO ANN — SIEMPRE VISIBLE
    # ===========================================================
    st.subheader("🔍 Módulo ANN — EMERREL / EMERAC")

    if modelo_ann is None:
        st.info("ANN no disponible: revise archivos de pesos (IW.npy, bias_IW.npy, LW.npy, bias_out.npy).")
    elif df_meteo is None:
        st.info("No se pudo generar el DataFrame meteorológico limpio para la ANN.")
    elif not all(c in df_meteo.columns for c in ["JD", "TMAX", "TMIN", "Prec"]):
        st.info("No se identificaron correctamente JD/TMAX/TMIN/Prec en el archivo cargado.")
    else:
        df_ann = df_meteo.copy().sort_values("JD")
        dias = df_ann["JD"].to_numpy()
        X_ann = df_ann[["JD","TMAX","TMIN","Prec"]].to_numpy(float)

        try:
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
            dias_ann = dias
            emerac_ann = emerac
            vals_ann = _compute_jd_percentiles(dias_ann, emerac_ann)

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

            # Descarga serie ANN
            csv_ann = df_ann.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar EMERREL/EMERAC simulada",
                csv_ann,
                "emergencia_simulada_ANN.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error ejecutando ANN: {e}")
            ann_ok = False

    # ===========================================================
    # 🔹 MÓDULO COMPARACIÓN CON CURVA OBSERVADA (RMSE + GRÁFICOS)
    # ===========================================================
    st.subheader("📊 Módulo comparación ANN vs curva observada")

    if not ann_ok or dias_ann is None or emerac_ann is None:
        st.info("Primero se debe generar correctamente la curva ANN (módulo anterior).")
        emerac_obs_interp = None
    else:
        archivo_obs = st.file_uploader(
            "Cargar curva observada (JD + EMERAC o JD + EMERREL):",
            key="obs", type=["csv","xlsx"]
        )

        emerac_obs_interp = None

        if archivo_obs is None:
            st.info("Cargue una curva observada para habilitar RMSE y gráficos comparativos.")
        else:
            try:
                if archivo_obs.name.endswith(".csv"):
                    df_obs = pd.read_csv(archivo_obs)
                else:
                    df_obs = pd.read_excel(archivo_obs)

                col_jd = None
                for k in ["jd","julian","dia"]:
                    for c in df_obs.columns:
                        if k in c.lower():
                            col_jd = c
                            break
                    if col_jd:
                        break

                col_emerac  = None
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

                    jd_model = np.array(dias_ann, float)
                    emerac_pred = np.array(emerac_ann, float)

                    emerac_obs_interp = np.interp(jd_model, jd_obs, emerac_obs)

                    # RMSE
                    if emerac_pred.max() > 0:
                        y_pred_norm = emerac_pred / emerac_pred.max()
                    else:
                        y_pred_norm = emerac_pred

                    if emerac_obs_interp.max() > 0:
                        y_obs_norm = emerac_obs_interp / emerac_obs_interp.max()
                    else:
                        y_obs_norm = emerac_obs_interp

                    rmse_norm = rmse_curvas(y_pred_norm, y_obs_norm)
                    rmse_raw  = rmse_curvas(emerac_pred, emerac_obs_interp)

                    st.markdown("### 📐 RMSE entre ANN y curva observada")
                    st.write(f"- **RMSE normalizado (0–1):** `{rmse_norm:.5f}`")
                    st.write(f"- **RMSE crudo:** `{rmse_raw:.5f}`  (si ambas curvas están en escala comparable)")

                    # Gráfico superpuesto
                    st.markdown("### 📈 Curvas superpuestas — ANN vs Observada")
                    fig_super = plot_comparativo_curvas(
                        jd_model,
                        emerac_pred,
                        emerac_obs_interp,
                        nombre_obs="Curva observada"
                    )
                    st.pyplot(fig_super)

                    # Comparativo visual
                    st.markdown("### 🎨 Comparativo visual profesional")
                    perc_pred = _compute_jd_percentiles(jd_model, emerac_pred)
                    perc_obs  = _compute_jd_percentiles(jd_model, emerac_obs_interp)

                    if perc_pred is not None and perc_obs is not None:
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
                        st.info("No se pudieron calcular percentiles para alguna de las curvas.")
            except Exception as e:
                st.error(f"Error procesando la curva observada: {e}")
                emerac_obs_interp = None

    # ===========================================================
    # 🔹 MÓDULO COMPARACIÓN CON PATRÓN MÁS CERCANO (CENTROIDES)
    # ===========================================================
    st.subheader("📌 Módulo comparación con patrón más cercano (centroides)")

    if not ann_ok or dias_ann is None or emerac_ann is None or vals_ann is None:
        st.info("No hay percentiles válidos de la curva ANN. Revise el módulo ANN.")
    else:
        try:
            cent = joblib.load(BASE/"predweem_model_centroides.pkl")
            C = cent["centroides"]   # index = patrones, cols = JD25..JD95

            v_ann = np.array(vals_ann)  # [d25,d50,d75,d95]
            dists = np.linalg.norm(C.values - v_ann, axis=1)
            idx_min = int(np.argmin(dists))
            patron_cercano = C.index[idx_min]
            dist_min = float(dists[idx_min])

            st.write(f"- **Patrón más cercano (centroides):** `{patron_cercano}`")
            st.write(f"- **Distancia en espacio (JD25–95):** `{dist_min:.2f}`")

            # Reconstruir curva acumulada del patrón a partir de centroides
            jd25 = float(C.iloc[idx_min]["JD25"])
            jd50 = float(C.iloc[idx_min]["JD50"])
            jd75 = float(C.iloc[idx_min]["JD75"])
            jd95 = float(C.iloc[idx_min]["JD95"])

            x_pts = np.array([jd25, jd50, jd75, jd95])
            y_pts = np.array([0.25, 0.50, 0.75, 1.00])

            emerac_pat = np.interp(dias_ann, x_pts, y_pts,
                                   left=0.0, right=1.0)

            emerac_ann_arr = np.array(emerac_ann, float)
            if emerac_ann_arr.max() > 0:
                emerac_ann_norm = emerac_ann_arr / emerac_ann_arr.max()
            else:
                emerac_ann_norm = emerac_ann_arr

            rmse_pat = rmse_curvas(emerac_ann_norm, emerac_pat)

            st.write(f"- **RMSE normalizado ANN vs patrón `{patron_cercano}`:** `{rmse_pat:.4f}`")

            fig_pat = plot_comparativo_curvas(
                dias_ann,
                emerac_ann_arr,
                emerac_pat,
                nombre_obs=f"Patrón {patron_cercano} (centroide)"
            )
            st.markdown("### 📈 Curva ANN vs curva del patrón más cercano (reconstruida)")
            st.pyplot(fig_pat)

            # Año representativo del patrón (más cercano al centroide)
            curvas_hist = _load_curves_emereac()
            if curvas_hist:
                labels_df = _assign_labels_from_centroids(curvas_hist)
                subset = labels_df[labels_df["patron"] == patron_cercano].copy()
                if not subset.empty:
                    Vcent = np.array([jd25, jd50, jd75, jd95])
                    mat = subset[["JD25","JD50","JD75","JD95"]].to_numpy(float)
                    d_rep = np.linalg.norm(mat - Vcent, axis=1)
                    idx_rep = int(np.argmin(d_rep))
                    anio_rep = int(subset.iloc[idx_rep]["anio"])
                    st.write(f"- **Año histórico representativo del patrón `{patron_cercano}`:** `{anio_rep}`")
                else:
                    st.info("No se encontraron años históricos asignados a ese patrón.")
            else:
                st.info("No se pudieron cargar las curvas históricas para identificar año representativo.")

        except Exception as e:
            st.error(f"No se pudo comparar con el patrón más cercano: {e}")

    # ===========================================================
    # 🔹 MÓDULO PERCENTILES ANN + RADAR VS PATRÓN
    # ===========================================================
    st.subheader("🎯 Módulo percentiles ANN + Radar vs patrón meteo")

    if not ann_ok or dias_ann is None or emerac_ann is None:
        st.info("No hay curva ANN válida para calcular percentiles.")
    else:
        vals = _compute_jd_percentiles(dias_ann, emerac_ann)
        if vals is None:
            st.info("No se pudieron calcular percentiles (d25–d95) sobre la curva ANN.")
        else:
            d25, d50, d75, d95 = vals
            st.write({
                "d25": round(d25, 1),
                "d50": round(d50, 1),
                "d75": round(d75, 1),
                "d95": round(d95, 1),
            })

            try:
                cent = joblib.load(BASE/"predweem_model_centroides.pkl")
                C = cent["centroides"]
                if 'patron_pred' in locals() and patron_pred in C.index:
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
                    st.info("El patrón predicho no se encuentra en los centroides o no se pudo determinar.")
            except Exception as e:
                st.error(f"No se pudo generar el radar comparativo: {e}")

    # ===========================================================
    # 🔹 MÓDULO CERTEZA DIARIA DEL PATRÓN (ANN + CENTROIDES)
    # ===========================================================
    st.subheader("📈 Módulo certeza diaria del patrón (ANN + centroides)")

    if not ann_ok or dias_ann is None or emerac_ann is None:
        st.info("No hay curva ANN válida para calcular certeza diaria.")
    else:
        try:
            cent = joblib.load(BASE/"predweem_model_centroides.pkl")
            C = cent["centroides"]

            if 'patron_pred' not in locals() or patron_pred not in C.index:
                st.warning("El patrón predicho no se encuentra en los centroides o no se pudo determinar.")
            else:
                df_ann_local = df_meteo.copy().sort_values("JD")
                fecha_col = None
                for c in df_ann_local.columns:
                    if "fecha" in c.lower() or "date" in c.lower():
                        fecha_col = c
                        break
                if fecha_col is not None:
                    df_ann_local[fecha_col] = pd.to_datetime(df_ann_local[fecha_col], errors="coerce")

                jd_eval = []
                fechas_eval = []
                probs_sel = []

                for i in range(4, len(dias_ann)):
                    jd_sub = dias_ann[:i+1]
                    emerac_sub = emerac_ann[:i+1]

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

                    if fecha_col is not None and fecha_col in df_ann_local.columns:
                        subf = df_ann_local[df_ann_local["JD"] == jd_sub[-1]]
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
                    JD_START = int(dias_ann.min())
                    JD_END   = int(dias_ann.max())
                    cobertura = (JD_END - JD_START + 1) / TEMPORADA_MAX

                    UMBRAL = 0.8
                    idx_crit = next((i for i, p in enumerate(probs_sel) if p >= UMBRAL), None)
                    idx_max  = int(np.argmax(probs_sel)) if len(probs_sel) > 0 else None

                    prob_crit = None
                    prob_max  = None
                    fecha_crit = None
                    fecha_max  = None

                    if idx_crit is not None:
                        prob_crit  = probs_sel[idx_crit]
                        fecha_crit = fechas_eval[idx_crit]

                    if idx_max is not None:
                        prob_max  = probs_sel[idx_max]
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

                    st.markdown("### 🧠 Resumen del momento crítico del patrón")

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

