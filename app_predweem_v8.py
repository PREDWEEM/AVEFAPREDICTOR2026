# ===============================================================
# 🌾 PREDWEEM v8 — AVEFA Predictor 2026
# Clasificación meteorológica → patrón (Early / Intermediate / Late / Extended)
# *** Sin modelos externos .pkl — ENTRENAMIENTO INTERNO ***
# Compatible con Streamlit Cloud (scikit-learn 1.7.2)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
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

st.title("🌾 AVEFA Predictor 2026 — PREDWEEM v8")
st.subheader("Clasificación meteorológica por patrón (Early / Intermediate / Late / Extended)")
st.info("🔧 El modelo meteo→patrón se entrena automáticamente dentro de la app usando sklearn disponible en Streamlit Cloud.")

# ===============================================================
# 🔵 ETAPA 1 — FUNCIONES SOBRE CURVAS DE EMERGENCIA ACUMULADA
# ===============================================================

def _compute_jd_percentiles(jd, emerac, qs=(0.25, 0.5, 0.75, 0.95)):
    """
    Calcula JD25, JD50, JD75, JD95 a partir de la curva de EMERAC (0-1).
    jd: vector de días julianos
    emerac: vector de emergencia acumulada (0-1)
    """
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
    return np.array(out, dtype=float)


def _load_curves_emereac():
    """
    Carga las curvas de EMERAC históricas 1977–1998 y 2000–2015
    desde los archivos interpolados.
    """
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
    Asigna a cada año un patrón (Early / Intermediate / Late / Extended)
    comparando JD25–JD95 con los centroides de predweem_model_centroides.pkl.
    """
    cent = joblib.load("predweem_model_centroides.pkl")
    C = cent["centroides"]  # DataFrame 4x4 con filas = patrones y columnas = JD25–JD95

    registros = []
    for year, df in sorted(curvas.items()):
        jd25, jd50, jd75, jd95 = _compute_jd_percentiles(df["JD"], df["EMERAC"])
        v = np.array([jd25, jd50, jd75, jd95])
        dists = ((C.values - v) ** 2).sum(axis=1) ** 0.5
        patron = C.index[np.argmin(dists)]
        registros.append(
            dict(
                anio=int(year),
                patron=str(patron),
                JD25=jd25,
                JD50=jd50,
                JD75=jd75,
                JD95=jd95,
            )
        )
    return pd.DataFrame(registros)

# ===============================================================
# 🔵 ETAPA 2 — FEATURES METEOROLÓGICAS (MISMOS DEL CLASIFICADOR VIEJO)
# ===============================================================

def _build_meteo_features_for_years(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye las features meteorológicas por año usando
    Bordenave_1977_2015_por_anio_con_JD.xlsx, con las mismas
    features del modelo viejo:

    - Tmin_mean
    - Tmax_mean
    - Tmed_mean
    - Prec_total
    - Prec_days_10mm
    - Tmed_FM (JD <= 121)
    - Prec_FM (JD <= 121)
    """
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
        df["Tmed"] = (df["TMIN"] + df["TMAX"]) / 2.0

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
# 🔵 ENTRENAMIENTO INTERNO DEL CLASIFICADOR (CACHEADO)
# ===============================================================

@st.cache_resource
def load_clf():
    """
    Entrena el clasificador meteo→patrón dentro de la app
    usando:
      - Curvas EMERAC históricas (JD25–95)
      - Centroides (Early/Intermediate/Late/Extended)
      - Meteo histórica 1977–2015

    y lo cachea para no reentrenar en cada interacción.
    """
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
# 🔵 ETAPA 3 — FEATURES A PARTIR DE UN DF METEO SUBIDO
# ===============================================================

def _build_features_from_df_meteo(df_meteo: pd.DataFrame) -> pd.DataFrame:
    """
    Construye UNA FILA de features a partir de un df_meteo cualquiera,
    intentando detectar automáticamente las columnas:

      - JD / Julian_days / dia_juliano / DOY
      - Tmin  (o Temperatura_Minima)
      - Tmax  (o Temperatura_Maxima)
      - Prec  (o Precipitacion_Pluviometrica / lluvia / ppt)

    y generando las mismas features que el modelo viejo.
    """
    df = df_meteo.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        """Detecta columnas permitiendo coincidencia parcial y minúsculas."""
        for n in names:
            n_low = n.lower()
            for col_low, col_orig in cols.items():
                if n_low == col_low or n_low in col_low:
                    return col_orig
        return None

    # detectar columnas
    c_jd   = pick("jd", "julian_days", "julianday", "dia_juliano", "doy")
    c_tmin = pick("tmin", "temperatura_minima", "temp_min")
    c_tmax = pick("tmax", "temperatura_maxima", "temp_max")
    c_prec = pick("prec", "precipitacion_pluviometrica", "precipitacion", "lluvia", "ppt", "prcp", "mm")

    if None in (c_jd, c_tmin, c_tmax, c_prec):
        raise ValueError(
            "No se identificaron correctamente JD/TMIN/TMAX/Prec.\n"
            f"Columnas detectadas: {list(df.columns)}\n"
            f"JD→{c_jd}, TMIN→{c_tmin}, TMAX→{c_tmax}, PRE→{c_prec}"
        )

    df["JD"]   = pd.to_numeric(df[c_jd], errors="coerce")
    df["TMIN"] = pd.to_numeric(df[c_tmin], errors="coerce")
    df["TMAX"] = pd.to_numeric(df[c_tmax], errors="coerce")
    df["Prec"] = pd.to_numeric(df[c_prec], errors="coerce")
    df = df.dropna(subset=["JD"])

    df["Tmed"] = (df["TMIN"] + df["TMAX"]) / 2.0

    feats = {
        "Tmin_mean": df["TMIN"].mean(),
        "Tmax_mean": df["TMAX"].mean(),
        "Tmed_mean": df["Tmed"].mean(),
        "Prec_total": df["Prec"].sum(),
        "Prec_days_10mm": (df["Prec"] >= 10).sum(),
    }

    sub = df[df["JD"] <= 121]  # ventana F/M confirmada
    feats["Tmed_FM"] = sub["Tmed"].mean()
    feats["Prec_FM"] = sub["Prec"].sum()

    return pd.DataFrame([feats])

# ===============================================================
# 🔵 ETAPA 4 — PREDICCIONES: 1 AÑO O MÚLTIPLES AÑOS
# ===============================================================

def predecir_patron(df_meteo: pd.DataFrame) -> dict:
    """
    Predice patrón para un solo dataset meteo (un año).
    """
    model = load_clf()
    Xnew = _build_features_from_df_meteo(df_meteo)
    proba = model.predict_proba(Xnew)[0]
    clases = model.classes_
    pred = clases[np.argmax(proba)]

    return {
        "clasificacion": str(pred),
        "probabilidades": dict(zip(map(str, clases), map(float, proba)))
    }


def predecir_patrones_multi_anio(df_meteo: pd.DataFrame) -> pd.DataFrame:
    """
    Si el archivo contiene múltiples años (columna Fecha),
    separa por año y predice un patrón por año.
    """
    df = df_meteo.copy()

    # Detectar columna de fecha
    col_fecha = None
    for c in df.columns:
        if "fecha" in c.lower():
            col_fecha = c
            break
    if col_fecha is None:
        raise ValueError("No se encontró columna tipo 'Fecha' para separar por año.")

    df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[col_fecha])

    years = sorted(df[col_fecha].dt.year.unique())

    resultados = []
    model = load_clf()

    for y in years:
        dfy = df[df[col_fecha].dt.year == y].copy()
        try:
            Xy = _build_features_from_df_meteo(dfy)
            proba = model.predict_proba(Xy)[0]
            clases = model.classes_
            pred = clases[np.argmax(proba)]

            row = {
                "Año": int(y),
                "Patrón": str(pred),
            }
            for c, p in zip(clases, proba):
                row[f"P_{c}"] = float(p)

            resultados.append(row)

        except Exception as e:
            resultados.append({
                "Año": int(y),
                "Patrón": f"ERROR: {e}",
            })

    return pd.DataFrame(resultados)

# ===============================================================
# 🔵 ETAPA 5 — INTERFAZ DE USUARIO
# ===============================================================

st.subheader("📤 Subir archivo meteorológico")
st.caption("Formato: CSV o Excel. Puede contener uno o varios años. "
           "Debe incluir JD + Tmin + Tmax + Prec (o nombres equivalentes) y opcionalmente Fecha.")

uploaded = st.file_uploader("Cargar archivo de meteorología:", type=["csv", "xlsx"])

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success("✅ Archivo cargado correctamente.")
        st.dataframe(df, use_container_width=True)

        # ¿Tiene columna Fecha?
        tiene_fecha = any("fecha" in c.lower() for c in df.columns)

        if tiene_fecha:
            st.subheader("🔍 Detección de múltiples años")
            tabla = predecir_patrones_multi_anio(df)
            st.success("Archivo con múltiples años procesado correctamente.")

            st.subheader("📊 Resultados de patrones por año")
            st.dataframe(tabla, use_container_width=True)

            st.download_button(
                "📥 Descargar tabla de patrones por año (CSV)",
                tabla.to_csv(index=False).encode("utf-8"),
                file_name="patrones_por_anio.csv",
                mime="text/csv"
            )
        else:
            st.subheader("🔎 Archivo interpretado como UN solo año")
            res = predecir_patron(df)
            st.markdown(f"### 🌱 Patrón predicho: **{res['clasificacion']}**")
            st.json(res["probabilidades"])

    except Exception as e:
        st.error(f"❌ Error procesando archivo: {e}")
else:
    st.info("⬆️ Subí un archivo para comenzar.")


