# ===============================================================
# 🌾 PREDWEEM v8.6 — AVEFA Predictor 2026 (Enhanced Version)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURACIÓN Y ESTILOS
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM v8.6 — AVEFA 2026", 
    layout="wide", 
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.stAppDeployButton {display: none;}

/* Custom styles */
.css-1d391kg {padding-top: 0rem;}
.stAlert {padding: 0.5rem;}
.metric-card {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #3b82f6;
}
.emergency-high { color: #ef4444; font-weight: bold; }
.emergency-medium { color: #f59e0b; font-weight: bold; }
.emergency-low { color: #10b981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
URL_METEO = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
CSV_OUT = BASE / "meteo_daily.csv"
MODEL_DIR = BASE / "models"
START_2026 = pd.Timestamp("2026-01-01")

# Crear directorios si no existen
MODEL_DIR.mkdir(exist_ok=True)

# ===============================================================
# 1. MOTOR DE ACTUALIZACIÓN MEJORADO
# ===============================================================
@st.cache_data(ttl=3600)  # Cache por 1 hora
def fetch_and_update_meteo():
    """Función mejorada para obtener datos meteorológicos con reintentos"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/xml, text/xml, */*",
        "Accept-Language": "es-ES,es;q=0.9",
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Obteniendo datos meteorológicos (intento {attempt + 1}/{max_retries})..."):
                r = requests.get(URL_METEO, headers=headers, timeout=20)
                r.raise_for_status()
                
                # Parse XML
                root = ET.fromstring(r.content)
                rows = []
                
                for d in root.findall(".//forecast/tabular/day"):
                    try:
                        fecha_elem = d.find("fecha")
                        tmax_elem = d.find("tmax")
                        tmin_elem = d.find("tmin")
                        precip_elem = d.find("precip")
                        
                        if all(elem is not None for elem in [fecha_elem, tmax_elem, tmin_elem, precip_elem]):
                            rows.append({
                                "Fecha": pd.to_datetime(fecha_elem.get("value")),
                                "TMAX": float(tmax_elem.get("value").replace(",", ".")),
                                "TMIN": float(tmin_elem.get("value").replace(",", ".")),
                                "Prec": float(precip_elem.get("value").replace(",", "."))
                            })
                    except (ValueError, AttributeError) as e:
                        st.warning(f"Error procesando un día: {e}")
                        continue
                
                if not rows:
                    st.error("No se encontraron datos válidos en el XML")
                    return None
                
                df_new = pd.DataFrame(rows)
                
                # Verificar si hay datos nuevos
                if CSV_OUT.exists():
                    df_old = pd.read_csv(CSV_OUT, parse_dates=["Fecha"])
                    
                    # Filtrar solo datos nuevos
                    last_date = df_old["Fecha"].max() if not df_old.empty else pd.Timestamp.min
                    df_new_filtered = df_new[df_new["Fecha"] > last_date]
                    
                    if not df_new_filtered.empty:
                        df_all = pd.concat([df_old, df_new_filtered], ignore_index=True)
                        st.success(f"✅ {len(df_new_filtered)} días nuevos agregados")
                    else:
                        df_all = df_old
                        st.info("ℹ️ No hay datos nuevos disponibles")
                else:
                    df_all = df_new
                    st.success(f"✅ {len(df_new)} días cargados inicialmente")
                
                # Cálculo robusto del Día Juliano 2026
                df_all["Fecha"] = pd.to_datetime(df_all["Fecha"])
                df_all["Juliano"] = (df_all["Fecha"] - START_2026).dt.days + 1
                df_all = df_all.sort_values("Fecha").drop_duplicates("Fecha")
                
                # Calcular variables derivadas
                df_all["TMEAN"] = (df_all["TMAX"] + df_all["TMIN"]) / 2
                df_all["Prec_Acum"] = df_all["Prec"].cumsum()
                
                # Guardar
                df_all.to_csv(CSV_OUT, index=False)
                return df_all
                
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                st.error(f"Error de conexión después de {max_retries} intentos: {e}")
                if CSV_OUT.exists():
                    st.warning("Usando datos almacenados localmente")
                    return pd.read_csv(CSV_OUT, parse_dates=["Fecha"])
                return None
            continue
        except Exception as e:
            st.error(f"Error inesperado: {e}")
            return None

# ===============================================================
# 2. MODELO ANN MEJORADO CON VALIDACIÓN
# ===============================================================
class EnhancedANNModel:
    def __init__(self, IW=None, bIW=None, LW=None, bLW=None):
        """Modelo ANN con validación y normalización robusta"""
        if all(x is not None for x in [IW, bIW, LW, bLW]):
            self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        else:
            # Valores por defecto si no hay archivos
            self.IW = np.random.randn(4, 10)
            self.bIW = np.random.randn(10)
            self.LW = np.random.randn(10)
            self.bLW = np.random.randn(1)
            
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([365, 41, 25.5, 84])
        
    def normalize(self, X):
        """Normalización robusta con verificación de límites"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Verificar límites
        for i in range(X.shape[1]):
            col = X[:, i]
            if np.any(col < self.input_min[i]) or np.any(col > self.input_max[i]):
                st.warning(f"⚠️ Algunos valores en la columna {i} están fuera del rango de entrenamiento")
        
        # Normalizar
        X_norm = 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1
        return X_norm
    
    def predict(self, X):
        """Predicción con manejo de errores"""
        try:
            Xn = self.normalize(X)
            emer = []
            for x in Xn:
                a1 = np.tanh(self.IW.T @ x + self.bIW)
                emer.append(np.tanh(self.LW @ a1 + self.bLW))
            
            emer = (np.array(emer).flatten() + 1) / 2
            emer = np.clip(emer, 0, 1)  # Asegurar entre 0 y 1
            emerac = np.cumsum(emer)
            emerac = emerac / emerac.max() if emerac.max() > 0 else emerac
            
            return emer, emerac
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            return np.zeros(len(X)), np.zeros(len(X))

@st.cache_resource
def load_resources():
    """Carga recursos con manejo de errores"""
    try:
        # Verificar archivos necesarios
        required_files = ["IW.npy", "bias_IW.npy", "LW.npy", "bias_out.npy"]
        missing_files = []
        
        for f in required_files:
            file_path = BASE / f
            if not file_path.exists():
                missing_files.append(f)
        
        if missing_files:
            st.warning(f"⚠️ Archivos de modelo faltantes: {', '.join(missing_files)}")
            st.info("Usando modelo por defecto (aleatorio)")
            ann = EnhancedANNModel()
        else:
            ann = EnhancedANNModel(
                np.load(BASE / "IW.npy"),
                np.load(BASE / "bias_IW.npy"),
                np.load(BASE / "LW.npy"),
                np.load(BASE / "bias_out.npy")
            )
        
        # Cargar centroides con manejo de errores
        centroids_path = BASE / "predweem_model_centroides.pkl"
        if centroids_path.exists():
            cent_data = joblib.load(centroids_path)
        else:
            st.warning("Archivo de centroides no encontrado")
            cent_data = {"centroides": pd.DataFrame({
                "JD25": [80, 90, 100],
                "JD50": [100, 110, 120],
                "JD75": [120, 130, 140],
                "JD95": [140, 150, 160]
            }, index=["Early", "Extended", "Late"])}
        
        return ann, cent_data
        
    except Exception as e:
        st.error(f"Error cargando recursos: {e}")
        return EnhancedANNModel(), {"centroides": pd.DataFrame()}

# ===============================================================
# 3. FUNCIONES UTILITARIAS MEJORADAS
# ===============================================================
def compute_percentiles(jd, emerac):
    """Cálculo robusto de percentiles"""
    if len(jd) == 0 or len(emerac) == 0:
        return None
    
    emer = np.asarray(emerac, dtype=float)
    jd = np.asarray(jd, dtype=float)
    
    if emer.max() <= 0:
        return None
    
    # Normalizar
    y = emer / emer.max()
    
    # Calcular percentiles
    percentiles = []
    for q in [0.25, 0.50, 0.75, 0.95]:
        try:
            if len(np.unique(y)) > 1:
                p = np.interp(q, y, jd)
            else:
                p = jd[len(jd)//2]  # Valor medio si no hay variación
            percentiles.append(p)
        except:
            percentiles.append(np.nan)
    
    return np.array(percentiles)

def classify_emergency_pattern(vals, centroids):
    """Clasificar patrón de emergencia"""
    if vals is None or centroids.empty:
        return "Desconocido", None
    
    try:
        # Calcular distancia a centroides
        dists = []
        for _, center in centroids.iterrows():
            dist = np.linalg.norm(center[["JD25", "JD50", "JD75", "JD95"]].values - vals)
            dists.append(dist)
        
        patterns = centroids.index.tolist()
        pattern = patterns[np.argmin(dists)]
        confidence = 1 / (1 + min(dists))  # Confianza basada en distancia
        
        return pattern, confidence
    except:
        return "Desconocido", None

# ===============================================================
# 4. INTERFAZ PRINCIPAL MEJORADA
# ===============================================================
def main():
    st.title("🌾 PREDWEEM v8.6 — Sistema Predictor de Emergencia de AVEFA 2026")
    st.markdown("---")
    
    # Barra lateral mejorada
    with st.sidebar:
        st.header("⚙️ Configuración del Modelo")
        
        # Parámetros del modelo
        col1, col2 = st.columns(2)
        with col1:
            smooth_win = st.slider("Ventana Suavizado", 1, 15, 3, 
                                  help="Número de días para suavizar la emergencia")
        with col2:
            umbral_alerta = st.slider("Umbral Alerta", 0.1, 1.0, 0.5, 0.05,
                                     help="Umbral para alertas de emergencia crítica")
        
        # Configuración de visualización
        st.subheader("📊 Visualización")
        show_heatmap = st.checkbox("Mostrar Mapa de Calor", True)
        show_radar = st.checkbox("Mostrar Radar de Percentiles", True)
        show_stats = st.checkbox("Mostrar Estadísticas", True)
        
        # Acciones
        st.subheader("🔄 Acciones")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Actualizar Datos", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("📊 Reporte Completo", use_container_width=True):
                st.session_state.show_report = True
        
        # Información del sistema
        st.markdown("---")
        st.subheader("ℹ️ Información")
        st.caption(f"Versión: 8.6")
        st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        if CSV_OUT.exists():
            file_size = CSV_OUT.stat().st_size / 1024
            st.caption(f"Datos: {file_size:.1f} KB")
    
    # Cargar recursos
    ann, cent_data = load_resources()
    
    # Obtener datos
    df_meteo = fetch_and_update_meteo()
    
    if df_meteo is None or df_meteo.empty:
        st.error("❌ No se pudieron cargar datos. Verifica la conexión o archivos locales.")
        return
    
    # Predicción
    X = df_meteo[["Juliano", "TMAX", "TMIN", "Prec"]].values
    emerrel, emerac = ann.predict(X)
    
    # Procesamiento robusto
    df_meteo["EMERREL"] = pd.Series(emerrel).rolling(window=smooth_win, center=True, min_periods=1).mean()
    df_meteo["EMERAC"] = emerac
    df_meteo["EMERAC_NORM"] = emerac / (emerac.max() if emerac.max() > 0 else 1)
    
    # ===============================================================
    # 5. PANEL DE CONTROL PRINCIPAL
    # ===============================================================
    
    # Métricas clave en la parte superior
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Días Analizados", len(df_meteo))
    
    with col2:
        current_emergency = df_meteo["EMERREL"].iloc[-1] if not df_meteo.empty else 0
        st.metric("🌱 Emergencia Actual", f"{current_emergency:.2%}")
    
    with col3:
        total_precip = df_meteo["Prec"].sum()
        st.metric("💧 Precip. Total", f"{total_precip:.1f} mm")
    
    with col4:
        avg_temp = df_meteo["TMEAN"].mean() if "TMEAN" in df_meteo.columns else 0
        st.metric("🌡️ Temp. Promedio", f"{avg_temp:.1f}°C")
    
    st.markdown("---")
    
    # ===============================================================
    # 6. VISUALIZACIONES PRINCIPALES
    # ===============================================================
    
    # A. MAPA SEMAFÓRICO (HEATMAP)
    if show_heatmap:
        st.subheader("🌡️ Mapa de Calor - Intensidad de Emergencia")
        
        # Crear matriz para heatmap
        heatmap_data = df_meteo[["Fecha", "EMERREL"]].copy()
        heatmap_data["Día"] = heatmap_data["Fecha"].dt.day
        heatmap_data["Mes"] = heatmap_data["Fecha"].dt.month
        heatmap_data["Año"] = heatmap_data["Fecha"].dt.year
        
        # Heatmap interactivo
        fig = px.density_heatmap(
            heatmap_data, 
            x="Fecha", 
            y="EMERREL",
            nbinsx=50,
            nbinsy=20,
            color_continuous_scale="RdYlGn_r",
            title="Distribución Temporal de Emergencia"
        )
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    # B. GRÁFICO DE PULSOS DE EMERGENCIA
    st.subheader("📈 Dinámica de Pulsos de Emergencia")
    
    fig = go.Figure()
    
    # Área de emergencia
    fig.add_trace(go.Scatter(
        x=df_meteo["Fecha"], 
        y=df_meteo["EMERREL"],
        fill='tozeroy',
        mode='lines',
        name='Emergencia Diaria',
        line=dict(color='#10b981', width=2),
        fillcolor='rgba(16, 185, 129, 0.3)'
    ))
    
    # Línea de acumulada
    fig.add_trace(go.Scatter(
        x=df_meteo["Fecha"], 
        y=df_meteo["EMERAC_NORM"],
        mode='lines',
        name='Emergencia Acumulada',
        line=dict(color='#3b82f6', width=2, dash='dash'),
        yaxis='y2'
    ))
    
    # Umbral de alerta
    fig.add_hline(
        y=umbral_alerta, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Umbral Alerta ({umbral_alerta})",
        annotation_position="bottom right"
    )
    
    # Alertas críticas
    alert_days = df_meteo[df_meteo["EMERREL"] > umbral_alerta]
    if not alert_days.empty:
        fig.add_trace(go.Scatter(
            x=alert_days["Fecha"],
            y=alert_days["EMERREL"],
            mode='markers',
            name='Alertas Críticas',
            marker=dict(color='red', size=10, symbol='triangle-up')
        ))
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        yaxis=dict(title="Emergencia Diaria", range=[0, 1]),
        yaxis2=dict(
            title="Emergencia Acumulada",
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ===============================================================
    # 7. DIAGNÓSTICO AGRONÓMICO
    # ===============================================================
    st.markdown("---")
    st.subheader("🔍 Diagnóstico Agronómico Avanzado")
    
    # Calcular percentiles
    vals = compute_percentiles(df_meteo["Juliano"], df_meteo["EMERAC"])
    
    if vals is not None:
        # Clasificar patrón
        pattern, confidence = classify_emergency_pattern(vals, cent_data["centroides"])
        
        # Layout de diagnóstico
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📊 Resultados Clave")
            
            # Tarjeta de patrón
            pattern_color = {
                "Early": "#ef4444",
                "Extended": "#f59e0b", 
                "Late": "#10b981",
                "Desconocido": "#6b7280"
            }.get(pattern, "#6b7280")
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: {pattern_color};">{pattern}</h4>
                <p>Patrón Detectado</p>
                {f'<small>Confianza: {confidence:.1%}</small>' if confidence else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas de percentiles
            st.markdown("#### 📅 Percentiles JD")
            for label, value in zip(["JD25", "JD50", "JD75", "JD95"], vals):
                st.metric(label, f"Día {int(value)}" if not np.isnan(value) else "N/A")
            
            # Recomendaciones según patrón
            st.markdown("#### 💡 Recomendaciones")
            recommendations = {
                "Early": "🚨 **ALERTA TEMPRANA**: Pico inminente. Aplicar residuales potentes inmediatamente.",
                "Extended": "⚠️ **EMERGENCIA EXTENDIDA**: Distribuir aplicaciones en múltiples momentos.",
                "Late": "✅ **PATRÓN TARDÍO**: Programar aplicaciones posteriores, monitorear continuamente.",
                "Desconocido": "🔍 **PATRÓN NO IDENTIFICADO**: Continuar monitoreo y validar datos."
            }
            st.info(recommendations.get(pattern, "Continuar monitoreo."))
        
        with col2:
            if show_radar:
                # Radar de percentiles comparativo
                st.markdown("### 📊 Comparativa de Percentiles")
                
                fig_radar = go.Figure()
                
                # Percentiles actuales
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals.tolist() + [vals[0]],
                    theta=['JD25', 'JD50', 'JD75', 'JD95', 'JD25'],
                    fill='toself',
                    name='Campaña 2026',
                    line=dict(color='#3b82f6', width=3)
                ))
                
                # Centroides de referencia
                for idx, (pattern_name, center) in enumerate(cent_data["centroides"].iterrows()):
                    fig_radar.add_trace(go.Scatterpolar(
                        r=center.tolist() + [center.iloc[0]],
                        theta=['JD25', 'JD50', 'JD75', 'JD95', 'JD25'],
                        fill='none',
                        name=f'Patrón {pattern_name}',
                        line=dict(color=['#ef4444', '#f59e0b', '#10b981'][idx % 3], 
                                 width=1, dash='dot')
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(vals.max() if not np.isnan(vals.max()) else 200, 200)]
                        ),
                        angularaxis=dict(direction="clockwise")
                    ),
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
    
    # ===============================================================
    # 8. PANEL DE DATOS Y EXPORTACIÓN
    # ===============================================================
    if show_stats:
        st.markdown("---")
        st.subheader("📋 Datos Detallados")
        
        # Pestañas para diferentes vistas de datos
        tab1, tab2, tab3 = st.tabs(["📈 Resumen Estadístico", "📅 Datos Diarios", "📤 Exportación"])
        
        with tab1:
            # Estadísticas descriptivas
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.markdown("**🌡️ Temperaturas**")
                temp_stats = df_meteo[["TMAX", "TMIN"]].describe()
                st.dataframe(temp_stats, use_container_width=True)
            
            with stats_col2:
                st.markdown("**💧 Precipitación**")
                prec_stats = df_meteo["Prec"].describe()
                st.dataframe(prec_stats, use_container_width=True)
            
            with stats_col3:
                st.markdown("**🌱 Emergencia**")
                emerg_stats = df_meteo[["EMERREL", "EMERAC"]].describe()
                st.dataframe(emerg_stats, use_container_width=True)
        
        with tab2:
            # Datos diarios con filtros
            col1, col2 = st.columns([1, 3])
            
            with col1:
                days_to_show = st.slider("Días a mostrar", 7, 90, 30)
                show_cols = st.multiselect(
                    "Columnas a mostrar",
                    df_meteo.columns.tolist(),
                    default=["Fecha", "TMAX", "TMIN", "Prec", "EMERREL"]
                )
            
            with col2:
                filtered_df = df_meteo[show_cols].tail(days_to_show)
                st.dataframe(
                    filtered_df.style.format({
                        "TMAX": "{:.1f}",
                        "TMIN": "{:.1f}",
                        "Prec": "{:.1f}",
                        "EMERREL": "{:.3f}"
                    }),
                    use_container_width=True,
                    height=400
                )
        
        with tab3:
            # Opciones de exportación
            st.markdown("### 📤 Exportar Datos")
            
            export_format = st.radio("Formato de exportación:", ["CSV", "Excel", "JSON"])
            
            if export_format == "CSV":
                csv_data = df_meteo.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv_data,
                    file_name="avefa_prediccion_2026.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                # Necesitarías pandas con soporte para Excel
                try:
                    excel_buffer = pd.ExcelWriter("temp.xlsx", engine='openpyxl')
                    df_meteo.to_excel(excel_buffer, index=False)
                    excel_buffer.close()
                    
                    with open("temp.xlsx", "rb") as f:
                        excel_data = f.read()
                    
                    st.download_button(
                        label="📥 Descargar Excel",
                        data=excel_data,
                        file_name="avefa_prediccion_2026.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error generando Excel: {e}")
            elif export_format == "JSON":
                json_data = df_meteo.to_json(orient="records", date_format="iso")
                st.download_button(
                    label="📥 Descargar JSON",
                    data=json_data,
                    file_name="avefa_prediccion_2026.json",
                    mime="application/json"
                )
            
            # Resumen para imprimir
            st.markdown("---")
            if st.button("🖨️ Generar Reporte Resumen"):
                with st.expander("📄 Reporte de Resumen", expanded=True):
                    st.markdown(f"""
                    ### 📋 Reporte PREDWEEM v8.6
                    
                    **Período analizado:** {df_meteo['Fecha'].min().date()} al {df_meteo['Fecha'].max().date()}
                    **Total de días:** {len(df_meteo)}
                    
                    **📊 Estadísticas clave:**
                    - Emergencia máxima: {df_meteo['EMERREL'].max():.2%}
                    - Emergencia promedio: {df_meteo['EMERREL'].mean():.2%}
                    - Días con alerta: {(df_meteo['EMERREL'] > umbral_alerta).sum()}
                    - Precipitación total: {df_meteo['Prec'].sum():.1f} mm
                    
                    **🔍 Diagnóstico:** {pattern}
                    **⚠️ Alertas activas:** {(df_meteo['EMERREL'] > umbral_alerta).any()}
                    
                    *Generado el {datetime.now().strftime('%Y-%m-%d %H:%M')}*
                    """)
    
    # ===============================================================
    # 9. PIE DE PÁGINA Y METADATOS
    # ===============================================================
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"📅 Última actualización: {datetime.now().strftime('%H:%M')}")
    
    with col2:
        st.caption(f"📊 Días en memoria: {len(df_meteo)}")
    
    with col3:
        if not alert_days.empty:
            st.error(f"⚠️ {len(alert_days)} días con alerta crítica")
        else:
            st.success("✅ Sin alertas críticas")

# ===============================================================
# EJECUCIÓN PRINCIPAL
# ===============================================================
if __name__ == "__main__":
    main()
