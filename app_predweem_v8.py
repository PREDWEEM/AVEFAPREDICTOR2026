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
    xls1 = pd.ExcelFile("emergencia_acumulada_interpolada 1977-1998.xlsx")
    for sh in xls1.sheet_names:
        df = pd.read_excel(xls1, sheet_name=sh)
        year = int(str(sh).split("_")[-1])
        curvas[year] = df[["JD", "EMERAC"]].copy()

    xls2 = pd.ExcelFile("emergencia_2000_2015_interpolada.xlsx")
    for sh in xls2.sheet_names:
        df = pd.read_excel(xls2, sheet_name=sh)
        year = int(str(sh).split("_")[-1])
        curvas[year] = df[["JD", "EMERAC"]].copy()

    return curvas


def _assign_labels_from_centroids(curvas):
    cent = joblib.load("predweem_model_centroides.pkl")
    C = cent["centroides"]

    registros = []
    for year, df in sorted(curvas.items()):
        jd25, jd50, jd75, jd95 = _compute_jd_percentiles(df["JD"], df["EMERAC"])
        v = np.array([jd25, jd50, jd75, jd95])
        dists = np.sqrt(((C.values - v) ** 2).sum(axis=1))
        patron = C.index[np.argmin(dists)]

        registros.append({
            "anio": year,
            "patron": str(patron),
            "JD25": jd25,
            "JD50": jd50,
            "JD75": jd75,
            "JD95": jd95,
        })
    return pd.DataFrame(registros)

# ===============================================================
# 🔵 ETAPA 2 — METEOROLOGÍA HISTÓRICA POR AÑO
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

        fm = df[df["JD"] <= 121]
        feats["Tmed_FM"] = fm["Tmed"].mean()
        feats["Prec_FM"] = fm["Prec"].sum()

        rows.append(feats)

    return pd.DataFrame(rows)

# ===============================================================
# 🔵 ENTRENAMIENTO CACHEADO DEL MODELO
# ===============================================================

@st.cache_resource
def load_clf():
    curvas = _load_curves_emereac()
    labels = _assign_labels_from_centroids(curvas)
    feat = _build_meteo_features_for_years(labels).dropna()

    X = feat[[
        "Tmin_mean", "Tmax_mean", "Tmed_mean",
        "Prec_total", "Prec_days_10mm",
        "Tmed_FM", "Prec_FM"
    ]]
    y = feat["patron"]

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=600,
            learning_rate=0.03,
            random_state=42
        )),
    ])
    clf.fit(X, y)
    return clf

# ===============================================================
# 🔵 FEATURES DESDE ARCHIVO CARGADO
# ===============================================================

def _build_features_from_df_meteo(df):
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            for k, v in cols.items():
                if n.lower() in k:
                    return v
        return None

    c_jd = pick("jd", "julian_days", "dia_juliano", "doy")
    c_tmin = pick("tmin", "temperatura_minima")
    c_tmax = pick("tmax", "temperatura_maxima")
    c_prec = pick("prec", "precipitacion", "lluvia", "ppt", "mm")

    if None in (c_jd, c_tmin, c_tmax, c_prec):
        raise ValueError("No se identificaron correctamente JD/TMIN/TMAX/Prec.")

    df2 = df.copy()
    df2["JD"] = pd.to_numeric(df2[c_jd], errors="coerce")
    df2["TMIN"] = pd.to_numeric(df2[c_tmin], errors="coerce")
    df2["TMAX"] = pd.to_numeric(df2[c_tmax], errors="coerce")
    df2["Prec"] = pd.to_numeric(df2[c_prec], errors="coerce")
    df2 = df2.dropna(subset=["JD"])

    df2["Tmed"] = (df2["TMIN"] + df2["TMAX"]) / 2

    feats = {
        "Tmin_mean": df2["TMIN"].mean(),
        "Tmax_mean": df2["TMAX"].mean(),
        "Tmed_mean": df2["Tmed"].mean(),
        "Prec_total": df2["Prec"].sum(),
        "Prec_days_10mm": (df2["Prec"] >= 10).sum(),
    }

    fm = df2[df2["JD"] <= 121]
    feats["Tmed_FM"] = fm["Tmed"].mean()
    feats["Prec_FM"] = fm["Prec"].sum()

    return pd.DataFrame([feats])

# ===============================================================
# 🔵 PREDICCIONES (1 AÑO O MULTI-AÑO)
# ===============================================================

def predecir_patron(df):
    model = load_clf()
    Xnew = _build_features_from_df_meteo(df)
    proba = model.predict_proba(Xnew)[0]
    clases = model.classes_
    pred = clases[np.argmax(proba)]
    return {
        "clasificacion": str(pred),
        "probabilidades": dict(zip(clases, proba))
    }


def predecir_patrones_multi_anio(df):
    # detectar columna Fecha
    col_fecha = next((c for c in df.columns if "fecha" in c.lower()), None)
    if col_fecha is None:
        raise ValueError("No se encontró columna Fecha.")

    df2 = df.copy()
    df2[col_fecha] = pd.to_datetime(df2[col_fecha], dayfirst=True, errors="coerce")
    df2 = df2.dropna(subset=[col_fecha])

    years = sorted(df2[col_fecha].dt.year.unique())
    model = load_clf()
    resultados = []

    for y in years:
        dfy = df2[df2[col_fecha].dt.year == y]
        try:
            Xy = _build_features_from_df_meteo(dfy)
            proba = model.predict_proba(Xy)[0]
            pred = model.classes_[np.argmax(proba)]
            row = {"Año": y, "Patrón": str(pred)}
            for clase, p in zip(model.classes_, proba):
                row[f"P_{clase}"] = float(p)
            resultados.append(row)
        except Exception as e:
            resultados.append({"Año": y, "Patrón": f"ERROR: {e}"})

    return pd.DataFrame(resultados)

# ===============================================================
# 🔵 INTERFAZ DE USUARIO
# ===============================================================

st.subheader("📤 Subir archivo meteorológico")
uploaded = st.file_uploader(
    "Cargar archivo de meteorología:",
    type=["csv", "xlsx"]
)

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success("Archivo cargado correctamente.")
        st.dataframe(df, use_container_width=True)

        # Detectar si tiene Fecha
        tiene_fecha = any("fecha" in c.lower() for c in df.columns)

        # =======================================================
        # 🟦 ARCHIVO MULTIANUAL (CON FECHA)
        # =======================================================
        if tiene_fecha:
            st.subheader("🔍 Detección de múltiples años")

            col_fecha = next(c for c in df.columns if "fecha" in c.lower())
            df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors="coerce")
            df = df.dropna(subset=[col_fecha])

            years = sorted(df[col_fecha].dt.year.unique())

            opcion = st.radio(
                "¿Qué deseás analizar?",
                ["Todos los años", "Seleccionar un año específico"]
            )

            # ------------------- TODOS LOS AÑOS -------------------
            if opcion == "Todos los años":
                tabla = predecir_patrones_multi_anio(df)
                st.subheader("📊 Resultados de patrones por año")
                st.dataframe(tabla, use_container_width=True)

                st.download_button(
                    "📥 Descargar tabla (CSV)",
                    tabla.to_csv(index=False).encode(),
                    "patrones_por_anio.csv"
                )

            # ------------------- UN SOLO AÑO -------------------
            else:
                year_sel = st.selectbox("Seleccionar año:", years)
                dfy = df[df[col_fecha].dt.year == year_sel]

                st.markdown(f"### 📅 Año seleccionado: **{year_sel}**")
                st.dataframe(dfy, use_container_width=True)

                res = predecir_patron(dfy)
                st.markdown(f"### 🌱 Patrón predicho: **{res['clasificacion']}**")
                st.json(res["probabilidades"])

                st.download_button(
                    f"📥 Descargar datos del año {year_sel}",
                    dfy.to_csv(index=False).encode(),
                    f"meteo_{year_sel}.csv"
                )

        # =======================================================
        # 🟩 ARCHIVO DE UN SOLO AÑO (SIN FECHA)
        # =======================================================
        else:
            st.subheader("🔎 Archivo interpretado como un solo año")

            res = predecir_patron(df)
            st.markdown(f"### 🌱 Patrón predicho: **{res['clasificacion']}**")
            st.json(res["probabilidades"])

    except Exception as e:
        st.error(f"❌ Error procesando archivo: {e}")

else:
    st.info("⬆️ Subí un archivo para comenzar.")
