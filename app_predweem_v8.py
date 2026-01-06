# ===============================================================
# 🌾 PREDWEEM v8.5 — AVEFA Predictor 2026 (Full Integrated)
# Actualización Automática + ANN + Radar + Diagnóstico Mixto
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURACIÓN Y RUTAS
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM v8.5 — AVEFA 2026", layout="wide")

# Ocultar menús de Streamlit
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
URL_METEO = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
CSV_OUT = BASE / "meteo_daily.csv"
START_2026 = pd.Timestamp("2026-01-01")

st.title("🌾 PREDWEEM v8.5 — AVEFA Predictor 2026")

# ===============================================================
# 1. MOTOR DE ACTUALIZACIÓN DE DATOS (MeteoBahia)
# ===============================================================
def fetch_and_update_meteo():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(URL_METEO, headers=headers, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        
        rows = []
        for d in root.findall(".//forecast/tabular/day"):
            f_str = d.find("fecha").get("value")
            f_dt = pd.to_datetime(f_str)
            
            rows.append({
                "Fecha": f_dt,
                "TMAX": float(d.find("tmax").get("value").replace(",", ".")),
                "TMIN": float(d.find("tmin").get("value").replace(",", ".")),
                "Prec": float(d.find("precip").get("value").replace(",", "."))
            })
        
        df_new = pd.DataFrame(rows)
        
        if CSV_OUT.exists():
            df_old = pd.read_csv(CSV_OUT, parse_dates=["Fecha"])
            df_all = pd.concat([df_old, df_new]).drop_duplicates("Fecha")
        else:
            df_all = df_new
            
        # Cálculo Robusto del Día Juliano 2026
        df_all["Fecha"] = pd.to_datetime(df_all["Fecha"])
        df_all["Juliano"] = (df_all["Fecha"] - START_2026).dt.days + 1
        
        df_all = df_all.sort_values("Fecha")
        df_all.to_csv(CSV_OUT, index=False)
        return df_all
    except Exception as e:
        st.error(f"Error actualizando datos de MeteoBahia: {e}")
        return pd.read_csv(CSV_OUT, parse_dates=["Fecha"]) if CSV_OUT.exists() else None

# ===============================================================
# 2. MODELO ANN Y FUNCIONES AUXILIARES
# ===============================================================
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([365, 41, 25.5, 84])

    def predict(self, X):
        Xn = 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1
        emer = []
        for x in Xn:
            a1 = np.tanh(self.IW.T @ x + self.bIW)
            emer.append(np.tanh(self.LW @ a1 + self.bLW))
        emer = (np.array(emer) + 1) / 2
        emerac = np.cumsum(emer)
        return np.diff(emerac, prepend=0), emerac

@st.cache_resource
def load_ann():
    return PracticalANNModel(
        np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"),
        np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy")
    )

def _compute_jd_percentiles(jd, emerac, qs=(0.25, 0.5, 0.75, 0.95)):
    emer = np.asarray(emerac, float)
    if emer.max() <= 0: return None
    y = emer / emer.max()
    return np.array([np.interp(q, y, jd) for q in qs], float)

# ===============================================================
# 3. SIDEBAR Y CARGA DE DATOS
# ===============================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    if st.button("🔄 Forzar Actualización Meteo"):
        st.cache_data.clear()
        st.rerun()
    smooth_win = st.slider("Suavizado (días)", 1, 9, 3)
    uploaded = st.file_uploader("O subir CSV manual", type=["csv"])

# Lógica de Prioridad de Datos
if uploaded:
    df_raw = pd.read_csv(uploaded, parse_dates=["Fecha"])
    st.info("Usando archivo manual.")
else:
    df_raw = fetch_and_update_meteo()

if df_raw is None:
    st.warning("No hay datos. Suba un archivo o verifique conexión.")
    st.stop()

# Asegurar columnas necesarias
df_meteo = df_raw.copy()
if "Juliano" not in df_meteo.columns:
    df_meteo["Juliano"] = (pd.to_datetime(df_meteo["Fecha"]) - START_2026).dt.days + 1

st.success(f"📅 Datos listos hasta: {df_meteo['Fecha'].max().date()} (Día Juliano: {df_meteo['Juliano'].max()})")

# ===============================================================
# 4. PROCESAMIENTO ANN V8.5
# ===============================================================
ann = load_ann()
X = df_meteo[["Juliano", "TMAX", "TMIN", "Prec"]].values
emerrel, emerac = ann.predict(X)

# Suavizado para visualización
df_meteo["EMERREL"] = pd.Series(emerrel).rolling(smooth_win, center=True).mean().fillna(0)
df_meteo["EMERAC"] = emerac
df_meteo["EMERAC_NORM"] = emerac / (emerac.max() if emerac.max() > 0 else 1)

# ===============================================================
# 5. CLASIFICACIÓN POR CENTROIDES Y REGLAS
# ===============================================================
cent_data = joblib.load(BASE / "predweem_model_centroides.pkl")
C = cent_data["centroides"]
vals = _compute_jd_percentiles(df_meteo["Juliano"], df_meteo["EMERAC"])

if vals is not None:
    d25, d50, d75, d95 = vals
    dists = np.linalg.norm(C[["JD25", "JD50", "JD75", "JD95"]].values - vals, axis=1)
    patron = C.index[np.argmin(dists)]
    
    # Probabilidades
    w = 1.0 / (dists + 1e-6)
    probs = w / w.sum()
    prob_dict = {str(C.index[i]): float(probs[i]) for i in range(len(C.index))}

    # --- VISUALIZACIÓN ---
    st.subheader(f"🟢 Patrón Dominante: {patron}")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(df_meteo["Juliano"], df_meteo["EMERREL"], color="red", lw=2)
        ax1.set_title("Tasa de Emergencia (EMERREL)")
        ax1.set_xlabel("Día Juliano 2026")
        st.pyplot(fig1)
        
    with col2:
        # Gráfico Radar
        labels_radar = ["JD25", "JD50", "JD75", "JD95"]
        angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False).tolist()
        angles += angles[:1]
        fig_rad, ax_r = plt.subplots(subplot_kw={'projection': 'polar'})
        val_plot = vals.tolist() + [vals[0]]
        ax_r.plot(angles, val_plot, color='black', lw=2, label="Actual")
        ax_r.fill(angles, val_plot, alpha=0.25)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(labels_radar)
        st.pyplot(fig_rad)

    # ===============================================================
    # 6. DIAGNÓSTICO AGRONÓMICO MIXTO (AAI)
    # ===============================================================
    st.divider()
    st.subheader("🧠 Análisis Agronómico Inteligente (AAI)")
    
    top_indices = np.argsort(probs)[::-1]
    p1, p2 = C.index[top_indices[0]], C.index[top_indices[1]]
    
    # Lógica de diagnóstico simplificada del motor v8.5
    if p1 == "Early":
        st.error("🚨 FOCO EN BARBECHO: Emergencia explosiva inminente. Use residuales potentes YA.")
    elif p1 == "Extended":
        st.warning("⚠️ ESTRATEGIA SOLAPADA: Múltiples cohortes. Requiere refuerzo de residuales a los 30 días.")
    elif p1 == "Late":
        st.info("🔵 FOCO EN POST-EMERGENCIA: El pico será tardío. No agote los residuales demasiado pronto.")
    
    st.write(f"**Combinación Probable:** {p1} ({probs[top_indices[0]]:.1%}) + {p2} ({probs[top_indices[1]]:.1%})")

else:
    st.error("Emergencia insuficiente para clasificar patrón.")

# Descarga de datos
st.download_button("📥 Descargar Datos 2026", df_meteo.to_csv(index=False), "avefa_2026_results.csv")
