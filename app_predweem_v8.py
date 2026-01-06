
# ===============================================================
# 🌾 PREDWEEM v8.5 — AVEFA Predictor 2026 (Full Integrated)
# Actualización automática + ANN + Diagnóstico Pro
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from pathlib import Path

# --- CONFIGURACIÓN Y RUTAS ---
st.set_page_config(page_title="PREDWEEM v8.5 — AVEFA 2026", layout="wide")
BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
URL_METEO = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
CSV_OUT = BASE / "meteo_daily.csv"
START_2026 = datetime(2026, 1, 1)

# --- 1. MOTOR DE ACTUALIZACIÓN DE DATOS (MeteoBahia) ---
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
            
            # Cálculo del Día Juliano relativo al 01/01/2026
            juliano = (f_dt - START_2026).days + 1
            
            rows.append({
                "Fecha": f_dt,
                "Juliano": juliano,
                "TMAX": float(d.find("tmax").get("value").replace(",", ".")),
                "TMIN": float(d.find("tmin").get("value").replace(",", ".")),
                "Prec": float(d.find("precip").get("value").replace(",", "."))
            })
        
        df_new = pd.DataFrame(rows)
        
        if CSV_OUT.exists():
            df_old = pd.read_csv(CSV_OUT, parse_dates=["Fecha"])
            df_all = pd.concat([df_old, df_new]).drop_duplicates("Fecha").sort_values("Fecha")
        else:
            df_all = df_new
            
        df_all.to_csv(CSV_OUT, index=False)
        return df_all
    except Exception as e:
        st.error(f"Error actualizando datos de MeteoBahia: {e}")
        return pd.read_csv(CSV_OUT, parse_dates=["Fecha"]) if CSV_OUT.exists() else None

# --- 2. MODELO ANN ---
class PracticalANNModel:
    def __init__(self):
        # Carga de archivos .npy (Deben estar en la misma carpeta)
        self.IW = np.load(BASE / "IW.npy")
        self.bIW = np.load(BASE / "bias_IW.npy")
        self.LW = np.load(BASE / "LW.npy")
        self.bLW = np.load(BASE / "bias_out.npy")
        self.input_min = np.array([1, 0, -7, 0])  # JD, Tmax, Tmin, Prec
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

# --- 3. INTERFAZ STREAMLIT ---
st.title("🌾 PREDWEEM v8.5 — AVEFA Predictor 2026")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    if st.button("🔄 Forzar Actualización Meteo"):
        st.cache_data.clear()
        st.rerun()
    smooth_win = st.slider("Suavizado (días)", 1, 7, 3)

# Carga de Datos
df_meteo = fetch_and_update_meteo()

if df_meteo is not None:
    st.success(f"📅 Última actualización: {df_meteo['Fecha'].max().date()} | Día Juliano: {df_meteo['Juliano'].max()}")
    
    # Ejecución de ANN
    ann = PracticalANNModel()
    X = df_meteo[["Juliano", "TMAX", "TMIN", "Prec"]].values
    emerrel, emerac = ann.predict(X)
    
    # Procesamiento para visualización
    df_meteo["EMERREL"] = emerrel
    df_meteo["EMERAC_NORM"] = emerac / (emerac.max() if emerac.max() > 0 else 1)
    
    # --- GRÁFICOS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Tasa de Emergencia")
        fig1, ax1 = plt.subplots()
        ax1.plot(df_meteo["Juliano"], df_meteo["EMERREL"].rolling(smooth_win).mean(), color="red")
        ax1.set_xlabel("Día Juliano (Inicio 01/01/2026)")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

    with col2:
        st.subheader("📊 Emergencia Acumulada")
        fig2, ax2 = plt.subplots()
        ax2.fill_between(df_meteo["Juliano"], df_meteo["EMERAC_NORM"], color="green", alpha=0.2)
        ax2.plot(df_meteo["Juliano"], df_meteo["EMERAC_NORM"], color="green", lw=2)
        ax2.set_xlabel("Día Juliano")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

    # --- DIAGNÓSTICO AGRONÓMICO ---
    st.divider()
    st.subheader("🧠 Diagnóstico de Manejo AVEFA")
    
    # Lógica de percentiles rápida
    jd50 = np.interp(0.5, df_meteo["EMERAC_NORM"], df_meteo["Juliano"])
    
    if jd50 < 110:
        st.error("🚨 PATRÓN EARLY DETECTADO: El pico de emergencia es inminente. Priorizar residuales potentes AHORA.")
    elif 110 <= jd50 <= 150:
        st.warning("⚠️ PATRÓN INTERMEDIATE: Emergencia escalonada. Monitorear re-brotes en 15 días.")
    else:
        st.info("🔵 PATRÓN LATE/EXTENDED: Inicio tardío. Los residuales de barbecho largo podrían agotarse.")

else:
    st.warning("No hay datos disponibles. Verifique conexión a MeteoBahia.")
