# ===============================================================
# 🌾 PREDWEEM v8 — AVEFA Predictor 2026 (Con ANN + Clasificación)
# - ENTRENAMIENTO INTERNO meteo→patrón usando centroides
# - ANN → EMERREL diaria + EMERAC acumulada
# - Percentiles d25–d95 (curva ANN) + Radar año vs patrón
# - Certeza diaria del patrón (probabilidad día a día)
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
    """
    Lee curvas EMERAC históricas desde:
    - emergencia_acumulada_interpolada 1977-1998.xlsx
    - emergencia_2000_2015_interpolada.xlsx
    Devuelve dict: year -> DataFrame[JD, EMERAC]
    """
    curvas = {}

    # 1977–1998
    xls1 = pd.ExcelFile(BASE / "emergencia_acumulada_interpolada 1977-1998.xlsx")
    for sh in xls1.sheet_names:
        df = pd.read_excel(xls1, sheet_name=sh)
        year = int(str(sh).split("_")[-1])
        curvas[year] = df[["JD", "EMERAC"]].copy()

    # 2000–2015
    xls2 = pd.ExcelFile(BASE / "emergencia_2000_2015_interpolada.xlsx")
    for sh in xls2.sheet_names:
        df = pd.read_excel(xls2, sheet_name=sh)
        year = int(str(sh).split("_")[-1])
        curvas[year] = df[["JD", "EMERAC"]].copy()

    return curvas


def _assign_labels_from_centroids(curvas):
    """
    Usa predweem_model_centroides.pkl para asignar patrón a cada año histórico
    en base a (JD25, JD50, JD75, JD95) de EMERAC.
    """
    cent = joblib.load(BASE / "predweem_model_centroides.pkl")
    C = cent["centroides"]   # DataFrame: index=patrones, cols=[JD25, JD50, JD75, JD95]

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

# ===============================================================
# 🔵 2. FEATURES METEOROLÓGICOS HISTÓRICOS (Bordenave 1977–2015)
# ===============================================================
def _build_meteo_features_for_years(labels_df):
    """
    A partir de las etiquetas (anio, patrón) y el archivo
    Bordenave_1977_2015_por_anio_con_JD.xlsx, construye
    un DataFrame con features meteorológicas y columna 'patron'.
    """
    xls = pd.ExcelFile(BASE / "Bordenave_1977_2015_por_anio_con_JD.xlsx")
    rows = []

    for _, row in labels_df.iterrows():
        year = int(row["anio"])
        patron = row["patron"]

        if str(year) not in xls.sheet_names:
            continue

        df = pd.read_excel(xls, sheet_name=str(year)).copy()

        # Normalizar nombres
        df.rename(columns={
            "Temperatura_Minima": "TMIN",
            "Temperatura_Maxima": "TMAX",
            "Precipitacion_Pluviometrica": "Prec",
        }, inplace=True)

        # Asegurar tipos numéricos
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

    # JD: priorizar Julian_days
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

    return pd.DataFrame([feats]), df  # devuelvo también df limpio para ANN y certeza diaria

# ===============================================================
# 🔵 4. ENTRENAMIENTO INTERNO CLASIFICADOR METEO → PATRÓN
# ===============================================================
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

    fig = plt.figure(figsize=(6,6))
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
    except Exception as e:
        st.error(f"Error en la clasificación meteorológica: {e}")
        st.stop()

    st.markdown(f"## 🌱 Patrón meteo→predicho: **{patron_pred}**")
    st.json(probs)

    # ===========================================================
    # ANN solo si tenemos modelo y columnas mínimas
    # ===========================================================
    if modelo_ann is not None:
        st.subheader("🔍 Emergencia simulada por ANN (EMERREL / EMERAC)")

        # df_meteo viene de predecir_patron(df_raw)
        if not all(c in df_meteo.columns for c in ["JD", "TMAX", "TMIN", "Prec"]):
            st.info("No se identificaron correctamente JD/TMAX/TMIN/Prec para ejecutar la ANN.")
        else:
            # ---------- 1) EMERGENCIA ANN ----------
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
                fig_er, ax_er = plt.subplots(figsize=(5,4))
                ax_er.plot(dias, emerrel_raw, label="EMERREL cruda (ANN)", color="red", alpha=0.6)
                ax_er.plot(dias, emerrel,     label="EMERREL procesada",   color="blue", linewidth=2)
                ax_er.set_xlabel("Día juliano")
                ax_er.set_ylabel("EMERREL (fracción diaria)")
                ax_er.set_title("EMERREL: ANN vs post-proceso")
                ax_er.legend()
                st.pyplot(fig_er)

            with col_ac:
                fig_ac, ax_ac = plt.subplots(figsize=(5,4))
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

            # ---------- 2) Percentiles ANN sobre EMERAC ----------
            st.subheader("📌 Percentiles ANN del año (sobre lo emergido)")
            vals = _compute_jd_percentiles(dias, emerac)
            if vals is not None:
                d25, d50, d75, d95 = vals
                st.write({
                    "d25": round(d25, 1),
                    "d50": round(d50, 1),
                    "d75": round(d75, 1),
                    "d95": round(d95, 1),
                })

                # ---------- 3) Radar vs patrón meteo predicho ----------
                st.subheader("🎯 Radar JD25–95: Año ANN vs patrón meteo")
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

            # ---------- 4) Certeza diaria del patrón (ANN + centroides) ----------
            st.subheader("📈 Certeza diaria del patrón (ANN + centroides)")

            try:
                cent = joblib.load(BASE / "predweem_model_centroides.pkl")
                C = cent["centroides"]   # index = patrones, cols = JD25, JD50, JD75, JD95

                if patron_pred not in C.index:
                    st.warning(f"El patrón predicho (**{patron_pred}**) no se encuentra en los centroides.")
                else:
                    jd_eval = []
                    fechas_eval = []
                    probs_sel = []

                    # Detectar posible columna de fecha en df_ann
                    fecha_col = None
                    for c in df_ann.columns:
                        if "fecha" in c.lower() or "date" in c.lower():
                            fecha_col = c
                            break
                    if fecha_col is not None:
                        df_ann[fecha_col] = pd.to_datetime(df_ann[fecha_col], errors="coerce")

                    # Recorremos día a día, usando la curva ANN truncada hasta cada día
                    for i in range(4, len(dias)):  # arrancamos un poco más adelante para evitar ruido
                        jd_sub = dias[:i+1]
                        emerac_sub = emerac[:i+1]

                        vals_i = _compute_jd_percentiles(jd_sub, emerac_sub)
                        if vals_i is None:
                            continue

                        v = np.array(vals_i)  # [d25, d50, d75, d95]

                        # Distancias a cada centroide
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
                        # Cobertura temporal (1-ene → 1-oct ~ JD 1–274)
                        TEMPORADA_MAX = 274
                        JD_START = int(dias.min())
                        JD_END   = int(dias.max())
                        cobertura = (JD_END - JD_START + 1) / TEMPORADA_MAX

                        # Momento crítico y máxima certeza
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

                        # ----- Gráfico -----
                        figp, axp = plt.subplots(figsize=(9,5))

                        # Eje X por fecha si se puede, si no por JD
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

                        # ----- Resumen textual -----
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

                        # ----- Nivel de confianza global -----
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

            # ---------- 5) Descarga de serie ANN ----------
            csv_ann = df_ann.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar EMERREL/EMERAC simulada",
                csv_ann,
                "emergencia_simulada_ANN.csv",
                mime="text/csv"
            )


