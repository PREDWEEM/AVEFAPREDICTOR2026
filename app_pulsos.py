# ===============================================================
# 🌧️ PREDWEEM — AVEFA Pulsos: alerta semanal de emergencia de
#    Avena fatua a partir de la lluvia ya registrada
# Modelo: Chantre et al. (Weed Research, enviado 2026)
# Datos: meteo_daily.csv (Bordenave, actualizado por GitHub Actions)
# ===============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import avefa_pulsos as ap

st.set_page_config(page_title="AVEFA Pulsos — alerta de emergencia", layout="wide", page_icon="🌧️")

BASE = Path(__file__).parent
CSV_REPO = BASE / "meteo_daily.csv"
PARAMS_FILE = BASE / "pulsos_params.json"
TZ = "America/Argentina/Buenos_Aires"

COLORS = {"alta": "#c2410c", "moderada": "#d97706", "baja": "#15803d",
          "lluvia": "#2563eb", "modelo": "#0f766e", "clima": "#94a3b8"}

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stAppDeployButton {display: none;}
div[data-testid="stMetricValue"] {font-size: 1.9rem;}
.alerta {padding: 0.9rem 1.1rem; border-radius: 0.6rem; margin: 0.4rem 0 1rem 0; color: white; font-size: 1.05rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Carga
# ---------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_repo_csv(mtime: float) -> pd.DataFrame:
    return ap.clean_meteo(pd.read_csv(CSV_REPO))


@st.cache_data(ttl=3600)
def source_summary(mtime: float) -> list[str]:
    """Resumen de fuentes de meteo_daily.csv (SIGA observado, ECMWF provisional, ECMWF ENS P50)."""
    raw = pd.read_csv(CSV_REPO, usecols=lambda c: c in {"Fecha", "TipoDato"})
    if "TipoDato" not in raw:
        return []
    raw["Fecha"] = pd.to_datetime(raw["Fecha"], errors="coerce")
    out = []
    labels = {"Observado": "Observado SIGA–INTA Bordenave",
              "Provisional": "Provisional ECMWF IFS (hasta que SIGA publique)",
              "Pronostico": "Pronóstico ECMWF ENS, mediana"}
    for tipo, lab in labels.items():
        f = raw.loc[raw["TipoDato"] == tipo, "Fecha"]
        if len(f):
            out.append(f"{lab}: {len(f)} días ({f.min():%d/%m} – {f.max():%d/%m})")
    return out


@st.cache_resource
def load_params(mtime: float) -> ap.Params:
    return ap.Params.load(PARAMS_FILE)


def today_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)


params = load_params(PARAMS_FILE.stat().st_mtime)

# ---------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Datos")
    fuente = st.radio("Fuente meteorológica",
                      ["Estación INTA Bordenave (SIGA + ECMWF)", "Subir un CSV propio"], index=0)
    upload = None
    if fuente.startswith("Subir"):
        upload = st.file_uploader("CSV con columnas Fecha, TMAX, TMIN, Prec", type=["csv"])
        st.caption("Datos diarios desde el 1 de enero de la campaña. Fecha en formato AAAA-MM-DD.")

    try:
        if upload is not None:
            meteo = ap.clean_meteo(pd.read_csv(upload))
        else:
            meteo = load_repo_csv(CSV_REPO.stat().st_mtime)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo leer el CSV: {e}")
        st.stop()

    first, last = meteo["Fecha"].min(), meteo["Fecha"].max()
    if upload is None:
        for line in source_summary(CSV_REPO.stat().st_mtime):
            st.caption(line)
    default_emit = min(today_local(), last)
    try:  # ?fecha=AAAA-MM-DD en la URL
        qp = st.query_params.get("fecha")
        if qp:
            default_emit = min(max(pd.Timestamp(qp), first), last)
    except Exception:  # noqa: BLE001
        pass
    emit = pd.Timestamp(st.date_input("Fecha de emisión del pronóstico", value=default_emit.date(),
                                      min_value=first.date(), max_value=last.date()))
    st.caption("La lluvia posterior a esta fecha se ignora: el pronóstico usa solo la lluvia ya registrada. "
               "La temperatura de los 7 días siguientes se toma del pronóstico del CSV o de la climatología.")

    st.header("Alerta")
    umbral = st.select_slider("Umbral de alerta (% de la emergencia de la campaña en 7 días)",
                              options=[2.0, 2.5, 3.0, 4.0, 5.0], value=3.0)
    st.caption("3 %: visitas en ~33 % de las semanas, detecta ~61 % de los flushes. "
               "2 %: visitas en ~57 % de las semanas, detecta ~82 % (validación 1977–2015).")

    st.header("Conteos de campo (opcional)")
    usar_conteos = st.checkbox("Corregir con la proporción ya emergida")
    F0_user = None
    if usar_conteos:
        F0_user = st.slider("Proporción estimada ya emergida a la fecha de emisión (%)", 0, 99, 50) / 100
        st.caption("Con conteos el pronóstico mejora (AUC 0,77 vs 0,75), pero exige estimar el total de la campaña.")

# ---------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------
season_year = emit.year  # la campaña va del 1 de febrero en adelante
meteo_ext = ap.extend_with_climatology(meteo, emit + pd.Timedelta(days=8))
sim = ap.simulate(meteo_ext, params, rain_cutoff=emit, season_year=season_year)
days = sim.days
fc = ap.rolling_forecast(sim, horizon=7)
fc = fc[fc["Fecha"] <= emit].copy()

if F0_user is not None and not fc.empty:
    i = fc.index[-1]
    t = days.index[days["Fecha"] == emit][0]
    dH = (sim.B[t + 7] - sim.B[t]) + sum(sim.P[r, t + 7] - sim.P[r, t] for r, u in enumerate(sim.rain_idx) if u <= t)
    fc.loc[i, "F_emitido"] = F0_user
    fc.loc[i, "dF_7d"] = (1 - F0_user) * (1 - np.exp(-dH))

thr = umbral / 100
fc["alerta"] = fc["dF_7d"].apply(lambda v: ap.classify_alert(v, thr_low=thr * 2 / 3, thr_high=thr))

# ---------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------
st.title("🌧️ AVEFA Pulsos — alerta semanal de emergencia de *Avena fatua*")
PS = params.summary()


def fmt(x: float, nd: int = 0) -> str:
    """Número con coma decimal."""
    return f"{x:.{nd}f}".replace(".", ",")


st.caption(f"Cada lluvia dispara emergencia con un rezago fijo en tiempo térmico de suelo "
           f"(mediana {PS['mediana']:.0f} °C d sobre {params.Tb:g} °C). "
           "Como ese rezago supera una semana casi todo el año, la lluvia ya caída alcanza para anticipar los próximos 7 días.")

if params.provisional:
    st.warning("**Parámetros provisorios.** Reconstruidos a partir de los valores publicados en el manuscrito "
               "(rezago, multiplicador de sequedad, respuesta estacional, Fig. 4), no del ajuste original. "
               "Reemplazar `pulsos_params.json` por el vector ajustado a 1977–2015.", icon="⚠️")
else:
    st.caption(f"Parámetros: {params.source} Ver detalle al pie.")

if fc.empty:
    st.info(f"La campaña {season_year} empieza el 1 de febrero. Todavía no hay pronósticos para la fecha elegida.")
    st.stop()

now = fc.iloc[-1]
nivel = now["alerta"]
texto = {
    "alta": "Alerta alta: se espera un flush. Programar una recorrida del lote esta semana.",
    "moderada": "Alerta moderada: emergencia por encima de lo habitual. Recorrer si hay tratamientos pendientes.",
    "baja": "Sin alerta: emergencia baja esperada en los próximos 7 días.",
}[nivel]
st.markdown(f"<div class='alerta' style='background:{COLORS[nivel]}'><b>{emit:%d/%m/%Y}</b> · {texto}</div>",
            unsafe_allow_html=True)
if now["F_emitido"] >= 0.97:
    st.info("La campaña está prácticamente terminada según la simulación (≥ 97 % emergido). "
            "Para revisar la temporada, elegí una fecha anterior en la barra lateral.")

# próximos flushes de lluvias recientes
ev = sim.events.copy()
ev = ev[(ev["Fecha"] <= emit) & (ev["Fecha"] > emit - pd.Timedelta(days=60))]
F_at = days.set_index("Fecha")["F"]
ev["emergencia_esperada"] = [
    (1 - F_at.get(f - pd.Timedelta(days=1), 0.0)) * (1 - np.exp(-h)) for f, h in zip(ev["Fecha"], ev["hazard_total"])
]
proximos = ev[(ev["fecha_mediana_flush"] > emit) & (ev["emergencia_esperada"] >= 0.01)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Emergencia esperada en 7 días", f"{100 * now['dF_7d']:.1f} %",
          help="Porcentaje del total de la campaña. Referencia sin lluvias: "
               f"{100 * now['dF_7d_base']:.1f} %.")
c2.metric("Emergencia acumulada estimada", f"{100 * now['F_emitido']:.0f} %",
          help="Simulada solo con clima desde el 1 de febrero, o la ingresada si se usan conteos.")
s_now = float(days.loc[days["Fecha"] == emit, "S"].iloc[0])
c3.metric("Agua en el suelo", f"{100 * s_now / params.W:.0f} %",
          help=f"Balde de {fmt(params.W, 1)} mm del modelo de suelo. La lluvia sobre suelo seco dispara hasta "
               f"{fmt(PS['sequedad'], 1)} veces más emergencia.")
if not proximos.empty:
    nxt = proximos.sort_values("fecha_mediana_flush").iloc[0]
    c4.metric("Próximo flush (mediana)", f"{nxt['fecha_mediana_flush']:%d/%m}",
              help=f"Por la lluvia del {nxt['Fecha']:%d/%m} ({nxt['Prec']:.0f} mm).")
else:
    c4.metric("Próximo flush (mediana)", "—", help="Ninguna lluvia reciente con respuesta esperada ≥ 1 %.")

# ---------------------------------------------------------------
# Gráfico 1: alerta a lo largo de la campaña
# ---------------------------------------------------------------
st.subheader("Pronóstico a 7 días emitido cada día")
camp = days[(days["Fecha"] >= pd.Timestamp(season_year, 2, 1)) & (days["Fecha"] <= emit)]
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=camp["Fecha"], y=camp["Prec"], name="Lluvia (mm)", marker_color=COLORS["lluvia"],
                     opacity=0.35), secondary_y=True)
fig.add_trace(go.Scatter(x=fc["Fecha"], y=100 * fc["dF_7d_base"], name="Sin lluvias (base)",
                         line=dict(color=COLORS["clima"], dash="dot")), secondary_y=False)
fig.add_trace(go.Scatter(x=fc["Fecha"], y=100 * fc["dF_7d"], name="Pronóstico 7 días",
                         line=dict(color=COLORS["modelo"], width=2.5), fill="tozeroy",
                         fillcolor="rgba(15,118,110,0.12)"), secondary_y=False)
alto = fc[fc["alerta"] == "alta"]
fig.add_trace(go.Scatter(x=alto["Fecha"], y=100 * alto["dF_7d"], mode="markers", name="Alerta alta",
                         marker=dict(color=COLORS["alta"], size=6)), secondary_y=False)
fig.add_hline(y=umbral, line_dash="dash", line_color=COLORS["alta"],
              annotation_text=f"umbral {umbral:g} %", annotation_position="top left")
fig.update_yaxes(title_text="% de la campaña en 7 días", secondary_y=False, rangemode="tozero")
fig.update_yaxes(title_text="Lluvia (mm)", secondary_y=True, showgrid=False, rangemode="tozero")
fig.update_layout(height=380, hovermode="x unified", margin=dict(t=20, b=10, l=10, r=10),
                  legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------
# Gráfico 2 y tabla
# ---------------------------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.subheader("Emergencia acumulada")
    camp_all = days[days["Fecha"] >= pd.Timestamp(season_year, 2, 1)]
    camp_all = camp_all[camp_all["Fecha"] <= max(emit, camp_all["Fecha"].min())]
    fig2 = go.Figure()
    clim = ap.climatological_curve(pd.DatetimeIndex(camp_all["Fecha"]))
    fig2.add_trace(go.Scatter(x=camp_all["Fecha"], y=100 * clim, name="Curva media 1977–2015",
                              line=dict(color=COLORS["clima"], dash="dash")))
    fig2.add_trace(go.Scatter(x=camp_all["Fecha"], y=100 * camp_all["F"], name="Simulada con el clima de la campaña",
                              line=dict(color=COLORS["modelo"], width=2.5)))
    fig2.update_layout(height=340, yaxis_title="% de la campaña", yaxis_range=[0, 100],
                       hovermode="x unified", margin=dict(t=10, b=10, l=10, r=10),
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, width="stretch")
    st.caption("La curva de la campaña es orientativa: en validación no predijo la fecha del 50 % mejor que la curva media. "
               "Su uso es la alerta de la semana, no el pronóstico de toda la campaña.")

with right:
    st.subheader("Lluvias de los últimos 60 días")
    if ev.empty:
        st.write("Sin lluvias registradas en los últimos 60 días.")
    else:
        tab = pd.DataFrame({
            "Fecha": ev["Fecha"].dt.strftime("%d/%m"),
            "mm": ev["Prec"].round(1),
            "Suelo": (100 * ev["S_prev"] / params.W).round(0).astype(int).astype(str) + " %",
            "Emerg.": (100 * ev["emergencia_esperada"]).round(1).astype(str) + " %",
            "Flush": ev["fecha_mediana_flush"].dt.strftime("%d/%m") + "–"
                     + ev["fecha_p90_flush"].dt.strftime("%d/%m").fillna("…"),
        }).iloc[::-1]
        st.dataframe(tab, hide_index=True, width="stretch", height=330)
        st.caption("Suelo: agua en el suelo antes de la lluvia. Emerg.: emergencia total esperada, en % de la campaña. "
                   "Flush: fechas en que se acumula la mediana y el percentil 90 del rezago.")

# ---------------------------------------------------------------
# Información del modelo
# ---------------------------------------------------------------
with st.expander("Cómo funciona el modelo y sus límites"):
    clim_days = ap.soil_model(ap.climatological_temps(pd.date_range("2001-01-01", "2001-12-31")), params)
    C_clim = clim_days["C_TT"].to_numpy()

    def lag_days(month: int) -> int:
        u = pd.Timestamp(2001, month, 15).dayofyear - 1
        return int(np.searchsorted(C_clim, C_clim[u] + PS["mediana"]) - u)

    st.markdown(f"""
**Modelo.** La emergencia de cada día es una tasa estacional de base más la suma de los pulsos de cada lluvia
de los 60 días previos. Cada pulso tiene tres componentes:

- **Cuándo**: un rezago gamma en tiempo térmico de suelo a 5 cm sobre {params.Tb:g} °C (mediana {PS['mediana']:.0f} °C d,
  percentil 90 {PS['p90']:.0f} °C d). Con temperaturas medias son unos {lag_days(2)} días en febrero y {lag_days(7)} en julio.
- **Cuánto según la fecha**: la respuesta por unidad de lluvia sube de febrero a principios de julio
  ({fmt(PS['julio_febrero'], 1)} veces, máximo en el día {PS['dia_pico']:.0f}) y baja en primavera,
  a medida que el banco sale de la dormición.
- **Cuánto según el suelo**: la lluvia que rehumedece un suelo seco dispara hasta {fmt(PS['sequedad'], 1)} veces más
  emergencia que sobre suelo lleno. No hay umbral mínimo de lluvia; la respuesta crece con el logaritmo del agua agregada al suelo.

**Desempeño publicado (pronóstico a 7 días solo con clima).** Reduce 15 % el error de la curva media y sube el AUC para detectar
semanas con flush de 0,60 a 0,75 (validación dejando un año afuera, 38 campañas); en cinco campañas independientes, AUC 0,62 → 0,78.

**Límites.**
- Predice proporciones de la emergencia de la campaña, no plantas por m².
- Un sitio (Bordenave), suelo sin labranza. En suelo removido identifica flushes pero no mejora la cantidad: recalibrar antes de usarlo así.
- A 14 días pierde buena parte de la habilidad y a 21 días ninguna.
- No predice bien la fecha del 50 % ni la curva de toda la campaña.
- Cerca del 28 % de los flushes ocurre sin lluvia reciente (sobre todo julio–agosto) y el modelo no los anticipa.
""")
    st.markdown("**Vector en uso frente al manuscrito**")
    st.dataframe(pd.DataFrame({
        "Valor": ["Rezago, mediana (°C d)", "Rezago, moda (°C d)", "Rezago, percentil 90 (°C d)",
                  "Multiplicador de sequedad", "Respuesta julio / febrero", "Día de máxima respuesta"],
        "Manuscrito": ["139", "111", "260", "4,2", "18–20", "190–194"],
        "Vector en uso": [fmt(PS["mediana"]), fmt(PS["moda"]), fmt(PS["p90"]), fmt(PS["sequedad"], 2),
                          fmt(PS["julio_febrero"], 1), fmt(PS["dia_pico"])],
    }), hide_index=True, width="stretch")
    st.caption(f"Parámetros: {params.source}")

st.sidebar.download_button(
    "📥 Descargar pronósticos (CSV)",
    fc.assign(dF_7d_pct=100 * fc["dF_7d"], F_pct=100 * fc["F_emitido"]).to_csv(index=False),
    f"avefa_pulsos_{emit:%Y%m%d}.csv",
)
st.sidebar.caption(f"Datos: {first:%d/%m/%Y} – {last:%d/%m/%Y} · {len(meteo)} días")
