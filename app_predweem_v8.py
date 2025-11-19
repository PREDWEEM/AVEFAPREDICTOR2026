# ===============================================================
# 🌾 PREDWEEM v8 — AVEFA Predictor 2026 (OPCIÓN C, REVISADA)
# Clasificación meteorológica → patrón (Early / Intermediate / Late / Extended)
# + ANN para EMERREL/EMERAC, percentiles y radar comparativo
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
st.set_page_config(page_title="Predicción de Emergencia AVEFA — PREDWEEM v8", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 AVEFA Predictor 2026 — PREDWEEM v8")
st.subheader("Clasificación meteorológica (Early / Intermediate / Late / Extended) + Emergencia simulada (ANN)")

st.info("""
🔧 El modelo meteo→patrón se entrena automáticamente dentro de la app usando scikit-learn 1.7.2.  
Además, se utiliza una ANN para simular la emergencia (EMERREL/EMERAC) y calcular percentiles d25–d95 y un radar comparativo año vs patrón.
""")

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 🔧 UTILIDAD SEGURA
# ===============================================================
def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

# ===============================================================
# 🔵 ETAPA 1 — CURVAS DE EMERAC + JD25–95 + CENTROIDES
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
        out.append(float(jd[idx[0]]) if len(idx) else float(jd[-1]))
    return np.array(out, dtype=float)


def _load_curves_emereac():
    curvas = {}

    # ---- 1977–1998 ----
    xls1 = pd.ExcelFile("emergencia_acumulada_interpolada 1977-1998.xlsx")
    for sh in xls1.sheet_names:
        df = pd.read_excel(xls1, sheet_name=sh)
        year = int(str(sh).split("_")[-1])
        curvas[year] = df[["JD", "EMERAC"]].copy()

    # ---- 2000–2015 ----
    xls2 = pd.ExcelFile("emergencia_2000_2015_interpolada.xlsx")
    for sh in xls2.sheet_names:
        df = pd.read_excel(xls2, sheet_name=sh)
        year = int(str(sh).split("_")[-1])
        curvas[year] = df[["JD", "EMERAC"]].copy()

    return curvas


def _assign_labels_from_centroids(curvas):
    """
    Usa predweem_model_centroides.pkl para asignar patrón (Early/Int/Late/Ext)
    a cada año histórico según JD25–95.
    """
    cent = joblib.load("predweem_model_centroides.pkl")
    C = cent["centroides"]  # DataFrame: index = patrones, cols = JD25..JD95

    registros = []
    for year, df in sorted(curvas.items()):
        jd25, jd50, jd75, jd95 = _compute_jd_percentiles(df["JD"], df["EMERAC"])
        v = np.array([jd25, jd50, jd75, jd95])
        dists = ((C.values - v) ** 2).sum(axis=1) ** 0.5
        patron = C.index[np.argmin(dists)]

        registros.append({
            "anio": int(year),
            "patron": str(patron),
            "JD25": jd25,
            "JD50": jd50,
            "JD75": jd75,
            "JD95": jd95
        })
    return pd.DataFrame(registros)


# ===============================================================
# 🔵 ETAPA 2 — FEATURES METEOROLÓGICAS (IGUAL AL SCRIPT ORIGINAL)
# ===============================================================

def _build_meteo_features_for_years(labels_df):
    xls = pd.ExcelFile("Bordenave_1977_2015_por_anio_con_JD.xlsx")
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


# ===============================================================
# 🔵 FEATURES PARA ARCHIVOS METEOROLÓGICOS NUEVOS (IGUAL AL ORIGINAL)
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

    c_jd   = pick("jd", "dia", "julian", "doy")
    c_tmin = pick("tmin", "temperatura_minima")
    c_tmax = pick("tmax", "temperatura_maxima")
    c_prec = pick("prec", "lluvia", "ppt", "prcp", "pluviometrica")

    if None in (c_jd, c_tmin, c_tmax, c_prec):
        raise ValueError("No se identificaron correctamente JD/TMIN/TMAX/Prec.")

    df["JD"]   = pd.to_numeric(df[c_jd], errors="coerce")
    df["TMIN"] = pd.to_numeric(df[c_tmin], errors="coerce")
    df["TMAX"] = pd.to_numeric(df[c_tmax], errors="coerce")
    df["Prec"] = pd.to_numeric(df[c_prec], errors="coerce")
    df = df.dropna(subset=["JD"])

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

    return pd.DataFrame([feats])


# ===============================================================
# 🔵 ENTRENAMIENTO INTERNO DEL CLASIFICADOR METEOROLÓGICO
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


# ===============================================================
# 🔵 PREDICCIÓN — 1 AÑO O MULTI-AÑO (METEO → PATRÓN)
# ===============================================================

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


def predecir_patrones_multi_anio(df_meteo):
    model = load_clf()

    # Detectar columna Fecha o Año
    col_fecha = next((c for c in df_meteo.columns if "fecha" in c.lower()), None)
    col_anio  = next((c for c in df_meteo.columns if c.lower() in ["año", "ano", "year"]), None)

    if col_fecha:
        df_meteo[col_fecha] = pd.to_datetime(df_meteo[col_fecha], dayfirst=True, errors="coerce")
        df_meteo = df_meteo.dropna(subset=[col_fecha])
        years = sorted(df_meteo[col_fecha].dt.year.unique())
    elif col_anio:
        df_meteo[col_anio] = pd.to_numeric(df_meteo[col_anio], errors="coerce")
        df_meteo = df_meteo.dropna(subset=[col_anio])
        years = sorted(df_meteo[col_anio].unique())
    else:
        # Sin fecha ni año → caso multianual (igual que el original)
        years = list(range(1977, 2016))

    resultados = []

    for y in years:
        if col_fecha:
            dfy = df_meteo[df_meteo[col_fecha].dt.year == y]
        elif col_anio:
            dfy = df_meteo[df_meteo[col_anio] == y]
        else:
            xls = pd.ExcelFile("Bordenave_1977_2015_por_anio_con_JD.xlsx")
            dfy = pd.read_excel(xls, sheet_name=str(y))

        try:
            Xy = _build_features_from_df_meteo(dfy)
            proba = model.predict_proba(Xy)[0]
            clases = model.classes_
            pred = clases[np.argmax(proba)]

            row = {"Año": int(y), "Patrón": str(pred)}
            for c, p in zip(clases, proba):
                row[f"P_{c}"] = float(p)
            resultados.append(row)

        except Exception as e:
            resultados.append({"Año": int(y), "Patrón": f"ERROR: {e}"})

    return pd.DataFrame(resultados)


# ===============================================================
# 🔵 ANN — MODELO DE EMERGENCIA (EMERREL / EMERAC)
# ===============================================================

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
        # rango de entrenamiento original
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        """
        Devuelve EMERREL cruda de la ANN y EMERAC cruda (cumsum).
        El post-procesamiento se hace por fuera.
        """
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer) + 1) / 2    # 0–1 (diario, crudo)
        emer_ac = np.cumsum(emer)          # acumulada cruda
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac


@st.cache_resource
def load_ann():
    IW  = np.load(BASE / "IW.npy")
    bIW = np.load(BASE / "bias_IW.npy")
    LW  = np.load(BASE / "LW.npy")
    bLW = np.load(BASE / "bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)


def postprocess_emergence(emerrel_raw,
                          smooth=True,
                          window=3,
                          clip_zero=True):
    """
    Toma EMERREL cruda de la ANN y devuelve:
    - emerrel_proc: EMERREL suavizada / recortada
    - emerac_proc : EMERAC acumulada (no forzada a terminar en 1)
    """
    emer = np.array(emerrel_raw, dtype=float)

    # 1) Recortar posibles negativos
    if clip_zero:
        emer = np.maximum(emer, 0.0)

    # 2) Suavizado por media móvil
    if smooth and len(emer) > 1 and window > 1:
        window = int(window)
        window = max(1, min(window, len(emer)))
        if window > 1:
            kernel = np.ones(window, dtype=float) / window
            emer = np.convolve(emer, kernel, mode="same")

    # 3) EMERAC acumulada
    emerac = np.cumsum(emer)

    return emer, emerac


def calc_percentiles_trunc(dias, emerac):
    """
    Calcula d25–d95 tomando como referencia el máximo disponible
    (curva potencialmente truncada).
    """
    emerac = np.asarray(emerac)
    dias = np.asarray(dias)
    if emerac.max() == 0:
        return None
    y = emerac / emerac.max()   # normaliza respecto a lo emergido hasta la fecha
    d25 = np.interp(0.25, y, dias)
    d50 = np.interp(0.50, y, dias)
    d75 = np.interp(0.75, y, dias)
    d95 = np.interp(0.95, y, dias)
    return d25, d50, d75, d95


def radar_multiseries(values_dict, labels, title):
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    colors = {
        "Año evaluado": "blue",
        "Patrón predicho": "green"
    }

    for name, vals in values_dict.items():
        vals2 = list(vals) + [vals[0]]
        c = colors.get(name, None)
        ax.plot(angles, vals2, lw=2.5, label=name, color=c)
        ax.fill(angles, vals2, alpha=0.15, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.1))

    return fig


# ===============================================================
# 🔵 CONTROLES DE POST-PROCESO ANN EN SIDEBAR
# ===============================================================
with st.sidebar:
    st.header("Ajustes de emergencia (ANN)")
    use_smoothing = st.checkbox("Suavizar EMERREL", value=True)
    window_size   = st.slider("Ventana de suavizado (días)", min_value=1, max_value=9, value=3, step=1)
    clip_zero     = st.checkbox("Recortar negativos a 0", value=True)
    st.markdown("---")
    st.write("Estos ajustes se aplican a la emergencia simulada por la ANN para el año seleccionado.")


# ===============================================================
# 🔵 INTERFAZ — SUBIR ARCHIVO + SELECTOR DE AÑO
# ===============================================================

st.subheader("📤 Subir archivo meteorológico")
uploaded = st.file_uploader("Cargar archivo:", type=["csv", "xlsx"])

modelo_ann = safe(load_ann, "Error cargando pesos ANN (IW.npy, bias_IW.npy, LW.npy, bias_out.npy). La parte de emergencia no estará disponible.")

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success("Archivo cargado correctamente.")
        st.dataframe(df, use_container_width=True)

        # Detectar columnas Fecha/Año
        col_fecha = next((c for c in df.columns if "fecha" in c.lower()), None)
        col_anio  = next((c for c in df.columns if c.lower() in ["año", "ano", "year"]), None)

        if col_fecha:
            df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors="coerce")
            df = df.dropna(subset=[col_fecha])
            years = sorted(df[col_fecha].dt.year.unique())

        elif col_anio:
            df[col_anio] = pd.to_numeric(df[col_anio], errors="coerce")
            df = df.dropna(subset=[col_anio])
            years = sorted(df[col_anio].unique())

        else:
            st.info("No se encontró columna Fecha ni Año. Interpretando archivo como multianual (1977–2015).")
            years = list(range(1977, 2016))

        opcion = st.radio("¿Qué deseás analizar?", [
            "Todos los años (solo patrón meteo)",
            "Seleccionar un año específico (patrón + emergencia simulada)"
        ])

        # -------------------------------------------------------
        # TODOS LOS AÑOS → SOLO CLASIFICACIÓN METEOROLÓGICA
        # -------------------------------------------------------
        if opcion.startswith("Todos"):
            tabla = predecir_patrones_multi_anio(df)
            st.subheader("📊 Resultados de patrones por año (meteo → patrón)")
            st.dataframe(tabla, use_container_width=True)

            st.download_button(
                "📥 Descargar tabla (CSV)",
                tabla.to_csv(index=False).encode("utf-8"),
                "patrones_por_anio_meteo.csv",
                mime="text/csv"
            )

        # -------------------------------------------------------
        # UN AÑO → CLASIFICACIÓN METEO + ANN EMERGENCIA
        # -------------------------------------------------------
        else:
            year_sel = st.selectbox("Seleccionar año:", years)

            # Subconjunto del año seleccionado
            if col_fecha:
                dfy = df[df[col_fecha].dt.year == year_sel].copy()
                dfy = dfy.sort_values(col_fecha)
            elif col_anio:
                dfy = df[df[col_anio] == year_sel].copy()
            else:
                xls = pd.ExcelFile("Bordenave_1977_2015_por_anio_con_JD.xlsx")
                dfy = pd.read_excel(xls, sheet_name=str(year_sel))

            st.write(f"### 📅 Año seleccionado: {year_sel}")
            st.dataframe(dfy, use_container_width=True)

            # ----- CLASIFICACIÓN METEOROLÓGICA PRINCIPAL -----
            res = predecir_patron(dfy)
            patron_pred = res["clasificacion"]
            probs = res["probabilidades"]

            st.markdown(f"## 🌱 Patrón meteo→predicho: **{patron_pred}**")
            st.json(probs)

            # ===================================================
            # 🔶 EMERGENCIA SIMULADA POR ANN (SI ESTÁ DISPONIBLE)
            # ===================================================
            if modelo_ann is not None:
                st.subheader("🔍 Emergencia simulada por ANN (EMERREL / EMERAC)")

                # Detectar columnas mínimas para ANN (simple, sin tocar el clasificador)
                cols = {c.lower(): c for c in dfy.columns}

                def pick(*names):
                    for n in names:
                        for col_low, col_orig in cols.items():
                            if n.lower() in col_low:
                                return col_orig
                    return None

                c_jd   = pick("jd", "dia", "julian", "doy")
                c_tmin = pick("tmin", "temperatura_minima")
                c_tmax = pick("tmax", "temperatura_maxima")
                c_prec = pick("prec", "lluvia", "ppt", "prcp", "pluviometrica")

                df_ann = dfy.copy()

                # Si no hay JD pero sí Fecha, generamos JD
                if c_jd is None:
                    col_fecha_local = pick("fecha", "date")
                    if col_fecha_local is not None:
                        df_ann[col_fecha_local] = pd.to_datetime(df_ann[col_fecha_local], dayfirst=True, errors="coerce")
                        df_ann = df_ann.dropna(subset=[col_fecha_local])
                        df_ann["JD"] = df_ann[col_fecha_local].dt.dayofyear
                        c_jd = "JD"

                if c_jd is None or None in (c_tmin, c_tmax, c_prec):
                    st.info("No se pudieron identificar JD/TMIN/TMAX/Prec correctamente para ejecutar la ANN.")
                else:
                    df_ann["JD"]   = pd.to_numeric(df_ann[c_jd], errors="coerce")
                    df_ann["TMIN"] = pd.to_numeric(df_ann[c_tmin], errors="coerce")
                    df_ann["TMAX"] = pd.to_numeric(df_ann[c_tmax], errors="coerce")
                    df_ann["Prec"] = pd.to_numeric(df_ann[c_prec], errors="coerce")
                    df_ann = df_ann.dropna(subset=["JD"])

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

                    # ----- Gráficos EMERREL / EMERAC -----
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
                            ax_ac.plot(dias, emerac_raw/emerac_raw[-1], label="EMERAC cruda (normalizada)", color="orange", alpha=0.6)
                        else:
                            ax_ac.plot(dias, emerac_raw, label="EMERAC cruda", color="orange", alpha=0.6)
                        if emerac[-1] > 0:
                            ax_ac.plot(dias, emerac/emerac[-1], label="EMERAC procesada (normalizada)", color="green", linewidth=2)
                        else:
                            ax_ac.plot(dias, emerac, label="EMERAC procesada", color="green", linewidth=2)
                        ax_ac.set_xlabel("Día juliano")
                        ax_ac.set_ylabel("EMERAC (0–1 relativo al período)")
                        ax_ac.set_title("EMERAC: ANN vs post-proceso")
                        ax_ac.legend()
                        st.pyplot(fig_ac)

                    # ----- Cobertura temporal -----
                    st.subheader("🗓️ Cobertura temporal de los datos (ANN)")
                    JD_START = int(dias.min())
                    JD_END   = int(dias.max())
                    TEMPORADA_MAX = 274  # 1-ene → 1-oct, aprox. temporada completa
                    cobertura = (JD_END - JD_START + 1) / TEMPORADA_MAX

                    st.write({
                        "JD inicio": JD_START,
                        "JD fin":    JD_END,
                        "Cobertura relativa de temporada (~1-ene a 1-oct)": f"{cobertura*100:.1f} %",
                    })

                    # ----- Percentiles d25–d95 sobre EMERAC simulada -----
                    resp = calc_percentiles_trunc(dias, emerac)
                    if resp is not None:
                        d25, d50, d75, d95 = resp
                        st.subheader("📌 Percentiles simulados del año (ANN, sobre lo emergido hasta la fecha)")
                        st.write({
                            "d25 (del período observado)": round(d25, 1),
                            "d50 (del período observado)": round(d50, 1),
                            "d75 (del período observado)": round(d75, 1),
                            "d95 (del período observado)": round(d95, 1)
                        })

                        # ----- Radar comparativo JD25–95: año vs patrón meteo predicho -----
                        st.subheader("🎯 Radar comparativo JD25–95: Año (ANN) vs patrón meteo predicho")

                        try:
                            cent = joblib.load("predweem_model_centroides.pkl")
                            C = cent["centroides"]  # DataFrame
                            if patron_pred in C.index:
                                vals_year = [d25, d50, d75, d95]
                                vals_pat  = list(C.loc[patron_pred][["JD25", "JD50", "JD75", "JD95"]].values)

                                fig_rad = radar_multiseries(
                                    {
                                        "Año evaluado": vals_year,
                                        "Patrón predicho": vals_pat
                                    },
                                    labels=["d25", "d50", "d75", "d95"],
                                    title=f"Radar — Año {year_sel} (ANN) vs patrón {patron_pred}"
                                )
                                st.pyplot(fig_rad)
                            else:
                                st.info("El patrón predicho no se encuentra en la tabla de centroides (predweem_model_centroides.pkl).")
                        except Exception as e:
                            st.error(f"No se pudo generar el radar comparativo: {e}")

                    # ----- Descarga de serie simulada -----
                    csv_ann = df_ann.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"📥 Descargar EMERREL/EMERAC simulada {year_sel}",
                        csv_ann,
                        f"emergencia_simulada_{year_sel}.csv",
                        mime="text/csv"
                    )

            # ----- Descarga de meteo del año -----
            st.download_button(
                f"📥 Descargar datos meteorológicos del año {year_sel}",
                dfy.to_csv(index=False).encode("utf-8"),
                f"meteo_{year_sel}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error procesando archivo: {e}")
else:
    st.info("⬆️ Subí un archivo para comenzar.")
