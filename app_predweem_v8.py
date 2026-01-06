
# ===============================================================
# 🌾 PREDWEEM v8.5 — AVEFA Predictor 2026 (Full Integrated)
# ANN + Diagnóstico Pro + Visualización de Pulsos Interactivos
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURACIÓN Y RUTAS
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM v8.5 — AVEFA 2026", layout="wide", page_icon="🌾")

# Estilos CSS para limpieza de interfaz
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
            rows.append({
                "Fecha": pd.to_datetime(f_str),
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
            
        df_all["Juliano"] = (pd.to_datetime(df_all["Fecha"]) - START_2026).dt.days + 1
        df_all = df_all.sort_values("Fecha")
        df_all.to_csv(CSV_OUT, index=False)
        return df_all
    except Exception as e:
        st.error(f"Error actualizando datos: {e}")
        return pd.read_csv(CSV_OUT, parse_dates=["Fecha"]) if CSV_OUT.exists() else None

# ===============================================================
# 2. MODELO ANN Y FUNCIONES TÉCNICAS
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
    ann = PracticalANNModel(
        np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"),
        np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy")
    )
    cent_data = joblib.load(BASE / "predweem_model_centroides.pkl")
    return ann, cent_data

def _compute_jd_percentiles(jd, emerac):
    emer = np.asarray(emerac, float)
    if emer.max() <= 0: return None
    y = emer / emer.max()
    return np.array([np.interp(q, y, jd) for q in [0.25, 0.5, 0.75, 0.95]], float)

# ===============================================================
# 3. INTERFAZ Y PROCESAMIENTO
# ===============================================================
st.title("🌾 PREDWEEM v8.5 — AVEFA Predictor 2026")

with st.sidebar:
    st.header("⚙️ Configuración")
    umbral_alerta = st.slider("Umbral de Alerta Crítica", 0.1, 1.0, 0.5)
    smooth_win = st.slider("Suavizado (días)", 1, 9, 3)
    if st.button("🔄 Actualizar Meteo"):
        st.cache_data.clear()
        st.rerun()

ann, cent_data = load_resources()
df_meteo = fetch_and_update_meteo()

if df_meteo is not None:
    # Predicción
    X = df_meteo[["Juliano", "TMAX", "TMIN", "Prec"]].values
    emerrel, emerac = ann.predict(X)
    df_meteo["EMERREL"] = pd.Series(emerrel).rolling(smooth_win, center=True).mean().fillna(emerrel)
    df_meteo["EMERAC"] = emerac
    
    # --- A. MAPA SEMAFÓRICO (HEATMAP) ---
    st.subheader("🌡️ Intensidad de Emergencia Diaria")
    # Escala de colores: Verde (Bajo) -> Amarillo (Medio) -> Rojo (Crítico)
    colorscale = [[0, "#dcfce7"], [0.4, "#16a34a"], [0.5, "#facc15"], [0.8, "#ef4444"], [1, "#b91c1c"]]
    fig_h = go.Figure(data=go.Heatmap(
        z=[df_meteo["EMERREL"]], x=df_meteo["Fecha"], y=["Intensidad"],
        colorscale=colorscale, zmin=0, zmax=1, showscale=True
    ))
    fig_h.update_layout(height=150, margin=dict(t=20, b=0, l=10, r=10))
    st.plotly_chart(fig_h, use_container_width=True)

    # --- B. DINÁMICA DE EMERGENCIA RELATIVA (PULSOS) ---
    st.subheader("📈 Monitoreo de Pulsos de Emergencia")
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(
        x=df_meteo["Fecha"], y=df_meteo["EMERREL"], 
        fill='tozeroy', line_color='#15803d', name="Tasa de Emergencia"
    ))
    fig_m.add_hline(y=umbral_alerta, line_dash="dash", line_color="red", 
                    annotation_text="Umbral de Alerta", annotation_position="top right")
    fig_m.update_layout(height=350, margin=dict(t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig_m, use_container_width=True)

    # --- C. DIAGNÓSTICO AVANZADO ---
    st.divider()
    vals = _compute_jd_percentiles(df_meteo["Juliano"], df_meteo["EMERAC"])
    
    if vals is not None:
        C = cent_data["centroides"]
        dists = np.linalg.norm(C[["JD25", "JD50", "JD75", "JD95"]].values - vals, axis=1)
        patron = C.index[np.argmin(dists)]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Patrón Detectado", patron)
            st.metric("JD50 (50% Emergencia)", f"Día {int(vals[1])}")
            
            # Lógica de alerta rápida
            if vals[1] < 120:
                st.error("🚨 ALERTA: Patrón Temprano detectado. El pico de emergencia es inminente.")
            elif patron == "Extended":
                st.warning("⚠️ PRECAUCIÓN: Emergencia prolongada. Se requiere solapamiento de residuales.")
            else:
                st.success("✅ Patrón estable. Siga el monitoreo de pulsos para aplicaciones post-emergentes.")

        with col2:
            # Gráfico Radar de Percentiles
            labels_radar = ["JD25", "JD50", "JD75", "JD95"]
            angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False).tolist() + [0]
            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(
                r=vals.tolist() + [vals[0]], theta=["JD25", "JD50", "JD75", "JD95", "JD25"],
                fill='toself', name='Campaña 2026', line_color='black'
            ))
            fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 300])), 
                                  showlegend=False, title="Radar de Percentiles (JD)")
            st.plotly_chart(fig_rad, use_container_width=True)

    # Exportar datos
    st.sidebar.download_button("📥 Descargar Reporte CSV", df_meteo.to_csv(index=False), "predweem_2026.csv")
else:
    st.warning("No se pudieron cargar datos meteorológicos.")
