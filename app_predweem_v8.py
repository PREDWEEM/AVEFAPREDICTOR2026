# ===============================================================
# 🌾 PREDWEEM v8.5 — AVEFA Predictor 2026 (Local CSV Mode)
# Carga exclusiva desde meteo_daily.csv
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURACIÓN Y ESTILOS
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM v8.5 — AVEFA 2026", layout="wide", page_icon="🌾")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
CSV_OUT = BASE / "meteo_daily.csv"
START_2026 = pd.Timestamp("2026-01-01")

# ===============================================================
# 1. MOTOR DE CARGA DE DATOS LOCALES
# ===============================================================
def load_local_meteo():
    """Carga los datos únicamente desde el archivo CSV local."""
    if not CSV_OUT.exists():
        return None
    
    try:
        df = pd.read_csv(CSV_OUT)
        # Asegurar formato de fecha
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        
        # Recálculo de Día Juliano para asegurar compatibilidad con el modelo
        df["Juliano"] = (df["Fecha"] - START_2026).dt.days + 1
        
        # Limpieza básica
        df = df.sort_values("Fecha").drop_duplicates("Fecha")
        return df
    except Exception as e:
        st.error(f"Error al leer meteo_daily.csv: {e}")
        return None

# ===============================================================
# 2. MODELO ANN Y LÓGICA TÉCNICA
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
        emer = (np.array(emer).flatten() + 1) / 2
        emerac = np.cumsum(emer)
        return np.diff(emerac, prepend=0), emerac

@st.cache_resource
def load_resources():
    # Nota: Asegúrate de que estos archivos .npy existan en la misma carpeta
    ann = PracticalANNModel(
        np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"),
        np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy")
    )
    return ann

# ===============================================================
# 3. INTERFAZ Y DASHBOARD
# ===============================================================
st.title("🌾 PREDWEEM v8.5 — AVEFA Predictor 2026")

with st.sidebar:
    st.header("⚙️ Configuración")
    umbral_alerta = st.slider("Umbral de Alerta Crítica", 0.1, 1.0, 0.5)
    smooth_win = st.slider("Ventana Suavizado (días)", 1, 9, 3)
    if st.button("🔄 Recargar CSV"):
        st.cache_data.clear()
        st.rerun()

# Carga de recursos y datos
try:
    ann = load_resources()
    df_meteo = load_local_meteo()
except FileNotFoundError as e:
    st.error(f"Faltan archivos de pesos del modelo (IW, LW, etc.): {e}")
    df_meteo = None

if df_meteo is not None:
    # Verificación de columnas necesarias
    required_cols = ["Juliano", "TMAX", "TMIN", "Prec"]
    if all(col in df_meteo.columns for col in required_cols):
        
        # Predicción ANN
        X = df_meteo[required_cols].values
        emerrel, emerac = ann.predict(X)
        
        # Procesamiento de resultados
        s_emerrel = pd.Series(emerrel)
        df_meteo["EMERREL"] = s_emerrel.rolling(window=smooth_win, center=True).mean().fillna(s_emerrel)
        df_meteo["EMERAC"] = emerac

        # --- A. MAPA SEMAFÓRICO (HEATMAP) ---
        st.subheader("🌡️ Intensidad de Emergencia Diaria")
        colorscale = [[0, "#dcfce7"], [0.4, "#16a34a"], [0.5, "#facc15"], [0.8, "#ef4444"], [1, "#b91c1c"]]
        fig_h = go.Figure(data=go.Heatmap(
            z=[df_meteo["EMERREL"]], x=df_meteo["Fecha"], y=["Intensidad"],
            colorscale=colorscale, zmin=0, zmax=1, showscale=True
        ))
        fig_h.update_layout(height=150, margin=dict(t=20, b=0, l=10, r=10))
        st.plotly_chart(fig_h, use_container_width=True)

        # --- B. DINÁMICA DE PULSOS ---
        st.subheader("📈 Monitoreo de Pulsos de Emergencia")
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=df_meteo["Fecha"], y=df_meteo["EMERREL"], 
            fill='tozeroy', line_color='#15803d', name="Tasa Diaria"
        ))
        fig_m.add_hline(y=umbral_alerta, line_dash="dash", line_color="red", 
                        annotation_text="Umbral de Alerta")
        fig_m.update_layout(height=350, margin=dict(t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig_m, use_container_width=True)

        # Sidebar: Info y Descarga
        st.sidebar.success(f"Datos cargados: {len(df_meteo)} días.")
        st.sidebar.download_button("📥 Descargar Reporte Completo", df_meteo.to_csv(index=False), "reporte_avefa_2026.csv")
    else:
        st.error(f"El CSV debe contener las columnas: {required_cols}")
else:
    st.info("📌 Por favor, asegúrate de que el archivo 'meteo_daily.csv' esté en la misma carpeta que este script.")
    st.warning("No se encontraron datos meteorológicos para procesar.")
