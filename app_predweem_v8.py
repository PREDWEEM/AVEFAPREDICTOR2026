# ===============================================================
# 🌾 PREDWEEM v8 — AVEFA Predictor 2026
# Clasificación meteorológica → patrón (Early / Intermediate / Late / Extended)
# *** ENTRENAMIENTO INTERNO — SIN MODELOS .PKL EXTERNOS ***
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
st.subheader("Clasificación meteorológica por patrón (Early / Intermediate / Late / Extended)")
st.info("🔧 El modelo meteo→patrón se entrena automáticamente dentro de la app usando scikit-learn 1.7.2 disponible en Streamlit Cloud.")


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
    cent = joblib.load("predweem_model_centroides.pkl")
    C = cent["centroides"]  # DataFrame

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
# 🔵 ETAPA 2 — FEATURES METEOROLÓGICAS (MISMAS DEL MODELO VIEJO)
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
# 🔵 ENTRENAMIENTO INTERNO DEL CLASIFICADOR
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
# 🔵 FEATURES PARA ARCHIVOS METEOROLÓGICOS NUEVOS
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
# 🔵 PREDICCIÓN — 1 AÑO O MULTI-AÑO
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
        df_meteo[col_fecha] = pd.to_datetime(df_meteo[col_fecha], dayfirst=True)
        years = sorted(df_meteo[col_fecha].dt.year.unique())
    elif col_anio:
        years = sorted(df_meteo[col_anio].unique())
    else:
        # Sin fecha ni año → caso multianual (usar archivo histórico)
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
# 🔵 INTERFAZ — SUBIR ARCHIVO + SELECTOR DE AÑO
# ===============================================================

st.subheader("📤 Subir archivo meteorológico")
uploaded = st.file_uploader("Cargar archivo:", type=["csv", "xlsx"])

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success("Archivo cargado correctamente.")
        st.dataframe(df, use_container_width=True)

        # Detectar columnas
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

        opcion = st.radio("¿Qué deseás analizar?", ["Todos los años", "Seleccionar un año específico"])

        if opcion == "Todos los años":
            tabla = predecir_patrones_multi_anio(df)
            st.subheader("📊 Resultados de patrones por año")
            st.dataframe(tabla, use_container_width=True)

            st.download_button(
                "📥 Descargar tabla (CSV)",
                tabla.to_csv(index=False).encode("utf-8"),
                "patrones_por_anio.csv",
                mime="text/csv"
            )

        else:
            year_sel = st.selectbox("Seleccionar año:", years)

            if col_fecha:
                dfy = df[df[col_fecha].dt.year == year_sel]
            elif col_anio:
                dfy = df[df[col_anio] == year_sel]
            else:
                xls = pd.ExcelFile("Bordenave_1977_2015_por_anio_con_JD.xlsx")
                dfy = pd.read_excel(xls, sheet_name=str(year_sel))

            st.write(f"### 📅 Año seleccionado: {year_sel}")
            st.dataframe(dfy, use_container_width=True)

            res = predecir_patron(dfy)
            st.markdown(f"### 🌱 Patrón predicho: **{res['clasificacion']}**")
            st.json(res["probabilidades"])

            st.download_button(
                f"📥 Descargar datos del año {year_sel}",
                dfy.to_csv(index=False).encode("utf-8"),
                f"meteo_{year_sel}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error procesando archivo: {e}")
else:
    st.info("⬆️ Subí un archivo para comenzar.")
