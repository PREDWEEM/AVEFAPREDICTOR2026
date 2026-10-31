# ===============================================================
# 🌧️ PREDWEEM — AVEFA Pulsos: alerta semanal de flushes de
#    Avena fatua (modelo de pulsos de lluvia + conteos de campo)
# Modelo: Chantre et al. (Weed Research, enviado 2026)
# Datos: meteo_daily.csv (Bordenave, SIGA + ECMWF, GitHub Actions)
# ===============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import avefa_pulsos as ap

st.set_page_config(page_title="AVEFA Pulsos — alerta de flushes", layout="wide", page_icon="🌧️")

BASE = Path(__file__).parent
CSV_REPO = BASE / "meteo_daily.csv"
PARAMS_FILE = BASE / "pulsos_params.json"
ALERT_FILE = BASE / "alerta_params.json"
TZ = "America/Argentina/Buenos_Aires"

ATENCION, ALERTA = 0.15, 0.25   # niveles fijos (validación 41 campañas, dejando una afuera)
COLORS = {"alerta": "#c2410c", "atencion": "#d97706", "baja": "#15803d",
          "lluvia": "#2a78d6", "modelo": "#0f766e", "clima": "#94a3b8", "conteo": "#6b7280"}
LEVEL_NAME = {"alerta": "Alerta", "atencion": "Atención", "baja": "Sin alerta"}
LEVEL_TEXT = {
    "alerta": "Flush probable: recorrer todo el lote esta semana y preparar el control.",
    "atencion": "Probabilidad de flush por encima de lo habitual: recorrer el lote si se está en la ventana de "
                "barbecho o presiembra, o si hay un tratamiento pendiente.",
    "baja": "Flush poco probable en los próximos 7 días.",
}

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stAppDeployButton {display: none;}
div[data-testid="stMetricValue"] {font-size: 1.8rem;}
.alerta {padding: 0.9rem 1.1rem; border-radius: 0.6rem; margin: 0.4rem 0 1rem 0; color: white; font-size: 1.05rem;}
</style>
""", unsafe_allow_html=True)


def fmt(x: float, nd: int = 0) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def today_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)


# ---------------------------------------------------------------
# Carga
# ---------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_repo_csv(mtime: float) -> pd.DataFrame:
    return ap.clean_meteo(pd.read_csv(CSV_REPO))


@st.cache_data(ttl=3600)
def source_summary(mtime: float) -> list[str]:
    raw = pd.read_csv(CSV_REPO, usecols=lambda c: c in {"Fecha", "TipoDato"})
    if "TipoDato" not in raw:
        return []
    raw["Fecha"] = pd.to_datetime(raw["Fecha"], errors="coerce")
    labels = {"Observado": "Observado SIGA–INTA Bordenave", "Provisional": "Provisional ECMWF IFS",
              "Pronostico": "Pronóstico ECMWF ENS (mediana)"}
    out = []
    for tipo, lab in labels.items():
        f = raw.loc[raw["TipoDato"] == tipo, "Fecha"]
        if len(f):
            out.append(f"{lab}: {len(f)} días ({f.min():%d/%m} – {f.max():%d/%m})")
    return out


@st.cache_resource
def load_models(m1: float, m2: float):
    return ap.Params.load(PARAMS_FILE), ap.AlertModel.load(ALERT_FILE)


params, alert = load_models(PARAMS_FILE.stat().st_mtime, ALERT_FILE.stat().st_mtime)

# ---------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Clima")
    fuente = st.radio("Fuente meteorológica", ["Estación INTA Bordenave (SIGA + ECMWF)", "Subir un CSV propio"], index=0)
    upload = None
    corregir = True
    if fuente.startswith("Subir"):
        upload = st.file_uploader("CSV con columnas Fecha, TMAX, TMIN, Prec", type=["csv"])
        corregir = st.checkbox("Es de la estación SIGA A872817 (aplicar corrección de temperatura)", value=False)
    try:
        meteo = ap.clean_meteo(pd.read_csv(upload)) if upload is not None else load_repo_csv(CSV_REPO.stat().st_mtime)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo leer el CSV: {e}")
        st.stop()
    if corregir and params.correccion_temperatura:
        meteo = ap.apply_temperature_correction(meteo, params.correccion_temperatura)
    first, last = meteo["Fecha"].min(), meteo["Fecha"].max()
    if upload is None:
        for line in source_summary(CSV_REPO.stat().st_mtime):
            st.caption(line)

    default_emit = min(today_local(), last)
    try:
        qp = st.query_params.get("fecha")
        if qp:
            default_emit = min(max(pd.Timestamp(qp), first), last)
    except Exception:  # noqa: BLE001
        pass
    emit = pd.Timestamp(st.date_input("Fecha de emisión", value=default_emit.date(),
                                      min_value=first.date(), max_value=last.date()))

    st.header("Conteos de campo")
    st.caption("Plántulas por marco en cada fecha de conteo, contadas y retiradas semanalmente. "
               "Solo importa la proporción entre fechas, así que vale cualquier tamaño de marco.")
    up_c = st.file_uploader("Subir planilla de conteos (Excel o CSV)", type=["xlsx", "xls", "xlsm", "csv"],
                            key="up_conteos",
                            help="Una fila por fecha de conteo y una columna por marco. Se busca la fila con "
                                 "el encabezado «Fecha»; títulos, promedios, acumulados y porcentajes se ignoran.")
    if up_c is not None:
        hoja = None
        if up_c.name.lower().endswith((".xlsx", ".xls", ".xlsm")):
            hojas = ap.excel_sheets(up_c)
            if len(hojas) > 1:
                hoja = st.selectbox("Hoja", hojas, key="hoja_conteos")
            up_c.seek(0)
        try:
            leido = ap.read_counts_file(up_c, hoja)
            cols = [c for c in leido.columns if c != "Fecha"]
            elegidas = st.multiselect("Columnas de marcos a usar", cols, default=cols, key="cols_conteos",
                                      help="Si la planilla tiene varios tratamientos (por ejemplo CR y SR), "
                                           "dejá solo los marcos del lote que querés evaluar.")
            if elegidas:
                st.session_state["conteos"] = leido[["Fecha"] + elegidas]
                st.caption(f"Leídas {leido['Fecha'].notna().sum()} fechas "
                           f"({leido['Fecha'].min():%d/%m/%Y} – {leido['Fecha'].max():%d/%m/%Y}).")
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer la planilla: {e}")
    base = st.session_state.get("conteos", pd.DataFrame({"Fecha": [None] * 3, "Marco 1": [None] * 3,
                                                          "Marco 2": [None] * 3, "Marco 3": [None] * 3}))
    base = base.copy()
    base["Fecha"] = ap.parse_dates(base["Fecha"]).dt.date
    edited = st.data_editor(base, num_rows="dynamic", width="stretch", key="editor_conteos",
                            column_config={"Fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY")})
    counts = ap.clean_counts(edited)
    if not counts.empty and (counts["Fecha"].max() < first or counts["Fecha"].min() > last):
        st.warning(f"Los conteos ({counts['Fecha'].min():%d/%m/%Y} – {counts['Fecha'].max():%d/%m/%Y}) no coinciden "
                   f"con el clima disponible ({first:%d/%m/%Y} – {last:%d/%m/%Y}); no se usan en la alerta.")
    if not counts.empty:
        st.download_button("📥 Guardar conteos (CSV)", edited.to_csv(index=False), "conteos_avefa.csv")

    st.header("Niveles de alerta")
    st.markdown(f"🟠 **Atención** desde {ATENCION:.0%} · 🔴 **Alerta** desde {ALERTA:.0%}")
    st.caption("Con conteos semanales (41 campañas, dejando una afuera): desde 15 % se marca el 48 % de las semanas "
               "y se cubre el 83 % de los flushes; desde 25 %, el 34 % de las semanas y el 73 % de los flushes. "
               "Por encima de 25 % cada aviso tiene el doble de chances de flush que una semana cualquiera; "
               "entre 15 y 25 %, más o menos las mismas. Usar Atención en la ventana de barbecho y presiembra, "
               "cuando perder un flush sale caro.")

# ---------------------------------------------------------------
# Modelo de pulsos
# ---------------------------------------------------------------
season_year = emit.year
meteo_ext = ap.extend_with_climatology(meteo, emit + pd.Timedelta(days=8))
sim = ap.simulate(meteo_ext, params, rain_cutoff=emit, season_year=season_year)
days = sim.days
fc = ap.rolling_forecast(sim, horizon=7)
fc = fc[fc["Fecha"] <= emit].copy()

st.title("🌧️ AVEFA Pulsos — alerta semanal de flushes de *Avena fatua*")
st.caption("Combina el modelo de pulsos de lluvia (cada lluvia dispara emergencia con un rezago en tiempo térmico de suelo) "
           "con los conteos de campo, que indican si la campaña viene adelantada o atrasada.")

if fc.empty:
    st.info(f"La campaña {season_year} empieza el 1 de febrero. Todavía no hay pronósticos para la fecha elegida.")
    st.stop()


def level(p: float) -> str:
    return "alerta" if p >= ALERTA else ("atencion" if p >= ATENCION else "baja")


# probabilidad solo con pulsos, todos los días
fc["prob_pulsos"] = [alert.probability(v) for v in fc["dF_7d"]]
pulso_now = float(fc["dF_7d"].iloc[-1])
sig = ap.counts_signal(counts, emit) if not counts.empty else None
usa_conteos = sig is not None and sig["dias_desde"] <= 7
prob = alert.probability(pulso_now, sig["g1"] if usa_conteos else None, bool(usa_conteos and sig["primer"]))
nivel = level(prob)

# historial de la alerta combinada en cada fecha de conteo
hist = []
for _, r in counts[(counts["Fecha"] <= emit) & (counts["Fecha"] >= pd.Timestamp(season_year, 2, 1))].iterrows():
    s = ap.counts_signal(counts, r["Fecha"])
    p_row = fc.loc[fc["Fecha"] == r["Fecha"], "dF_7d"]
    if s is None or p_row.empty:
        continue
    hist.append({"Fecha": r["Fecha"], "prob": alert.probability(float(p_row.iloc[0]), s["g1"], s["primer"]), "g1": s["g1"]})
hist = pd.DataFrame(hist)

st.markdown(f"<div class='alerta' style='background:{COLORS[nivel]}'><b>{emit:%d/%m/%Y}</b> · "
            f"<b>{LEVEL_NAME[nivel]}</b> · probabilidad de flush en los próximos 7 días: <b>{prob:.0%}</b>. "
            f"{LEVEL_TEXT[nivel]}</div>",
            unsafe_allow_html=True)
if not usa_conteos:
    if sig is None:
        st.info("Sin conteos cargados: la alerta usa solo el modelo de pulsos (menos precisa). "
                "Con conteos semanales la detección de flushes sube de ~63 % a ~72 % con el mismo esfuerzo de recorrida.")
    else:
        st.warning(f"El último conteo es del {sig['fecha']:%d/%m} ({sig['dias_desde']} días antes de la emisión): "
                   "la alerta usa solo el modelo de pulsos. Cargá el conteo de esta semana para combinarlos.")
if fc["F_emitido"].iloc[-1] >= 0.97:
    st.info("Según la simulación ya emergió ≥ 97 % de la campaña. Para revisar la temporada, elegí una fecha anterior.")

ev = sim.events.copy()
ev = ev[(ev["Fecha"] <= emit) & (ev["Fecha"] > emit - pd.Timedelta(days=60))]
F_at = days.set_index("Fecha")["F"]
ev["emergencia_esperada"] = [(1 - F_at.get(f - pd.Timedelta(days=1), 0.0)) * (1 - np.exp(-h))
                             for f, h in zip(ev["Fecha"], ev["hazard_total"])]
proximos = ev[(ev["fecha_mediana_flush"] > emit) & (ev["emergencia_esperada"] >= 0.01)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Probabilidad de flush (7 días)", f"{prob:.0%}",
          help="Flush: semana con ≥ 5 % de la emergencia de la campaña. Tasa histórica: "
               f"{alert.tasa_base:.0%} de las semanas.")
c2.metric("Emergencia esperada por lluvias", f"{fmt(100 * pulso_now, 1)} %",
          help="Pronóstico del modelo de pulsos para los próximos 7 días, en % de la campaña, con la lluvia ya registrada.")
c3.metric("Última semana / total contado",
          f"{sig['g1']:.0%}" if sig is not None else "—",
          help="Fracción de todo lo contado en la campaña que salió en el último intervalo. "
               "Alta = la emergencia viene activa." + (f" Conteo del {sig['fecha']:%d/%m}." if sig else ""))
if not proximos.empty:
    nxt = proximos.sort_values("fecha_mediana_flush").iloc[0]
    c4.metric("Próximo flush por lluvia (mediana)", f"{nxt['fecha_mediana_flush']:%d/%m}",
              help=f"Por la lluvia del {nxt['Fecha']:%d/%m} ({nxt['Prec']:.0f} mm).")
else:
    c4.metric("Próximo flush por lluvia (mediana)", "—", help="Ninguna lluvia reciente con respuesta esperada ≥ 1 %.")

# ---------------------------------------------------------------
# Gráfico: probabilidad, conteos y lluvia (paneles con eje propio)
# ---------------------------------------------------------------
st.subheader("Campaña: probabilidad de flush, conteos y lluvia")
camp = days[(days["Fecha"] >= pd.Timestamp(season_year, 2, 1)) & (days["Fecha"] <= emit)]
cc = counts[(counts["Fecha"] >= pd.Timestamp(season_year, 1, 1)) & (counts["Fecha"] <= emit)]
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.27, 0.23], vertical_spacing=0.05)
fig.add_trace(go.Scatter(x=fc["Fecha"], y=100 * fc["prob_pulsos"], name="Solo pulsos (diaria)",
                         line=dict(color=COLORS["modelo"], width=2)), row=1, col=1)
if not hist.empty:
    fig.add_trace(go.Scatter(x=hist["Fecha"], y=100 * hist["prob"], name="Pulsos + conteos", mode="lines+markers",
                             line=dict(color="#2a78d6", width=2),
                             marker=dict(size=9, color=[COLORS[level(v)] for v in hist["prob"]],
                                         line=dict(color="white", width=1))), row=1, col=1)
fig.add_hrect(y0=100 * ATENCION, y1=100 * ALERTA, fillcolor=COLORS["atencion"], opacity=0.08, line_width=0, row=1, col=1)
fig.add_hrect(y0=100 * ALERTA, y1=100, fillcolor=COLORS["alerta"], opacity=0.06, line_width=0, row=1, col=1)
fig.add_hline(y=100 * ATENCION, line_dash="dot", line_color=COLORS["atencion"], row=1, col=1,
              annotation_text=f"Atención ≥ {ATENCION:.0%}", annotation_position="bottom left")
fig.add_hline(y=100 * ALERTA, line_dash="dash", line_color=COLORS["alerta"], row=1, col=1,
              annotation_text=f"Alerta ≥ {ALERTA:.0%}", annotation_position="top left")
fig.add_trace(go.Bar(x=cc["Fecha"], y=cc["media"], name="Conteo (plántulas por marco)",
                     marker_color=COLORS["conteo"]), row=2, col=1)
fig.add_trace(go.Bar(x=camp["Fecha"], y=camp["Prec"], name="Lluvia (mm)", marker_color=COLORS["lluvia"]), row=3, col=1)
fig.update_yaxes(title_text="Prob. flush (%)", range=[0, 100], row=1, col=1)
fig.update_yaxes(title_text="Plántulas", rangemode="tozero", row=2, col=1)
fig.update_yaxes(title_text="Lluvia (mm)", rangemode="tozero", row=3, col=1)
fig.update_layout(height=560, hovermode="x unified", margin=dict(t=20, b=10, l=10, r=10),
                  legend=dict(orientation="h", y=1.06), bargap=0.15)
st.plotly_chart(fig, width="stretch")

left, right = st.columns([3, 2])
with left:
    st.subheader("Emergencia acumulada")
    camp_all = days[(days["Fecha"] >= pd.Timestamp(season_year, 2, 1)) & (days["Fecha"] <= emit)]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=camp_all["Fecha"], y=100 * ap.climatological_curve(pd.DatetimeIndex(camp_all["Fecha"])),
                              name="Curva media 1977–2015", line=dict(color=COLORS["clima"], dash="dash")))
    fig2.add_trace(go.Scatter(x=camp_all["Fecha"], y=100 * camp_all["F"], name="Simulada con el clima de la campaña",
                              line=dict(color=COLORS["modelo"], width=2.5)))
    fig2.update_layout(height=320, yaxis_title="% de la campaña", yaxis_range=[0, 100], hovermode="x unified",
                       margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig2, width="stretch")
    st.caption("Orientativa: la simulación solo con clima no anticipa si la campaña viene adelantada o atrasada; "
               "para eso están los conteos.")
with right:
    st.subheader("Lluvias de los últimos 60 días")
    if ev.empty:
        st.write("Sin lluvias registradas en los últimos 60 días.")
    else:
        tab = pd.DataFrame({
            "Fecha": ev["Fecha"].dt.strftime("%d/%m"), "mm": ev["Prec"].round(1),
            "Suelo": (100 * ev["S_prev"] / params.W).round(0).astype(int).astype(str) + " %",
            "Emerg.": (100 * ev["emergencia_esperada"]).round(1).astype(str) + " %",
            "Flush": ev["fecha_mediana_flush"].dt.strftime("%d/%m") + "–" + ev["fecha_p90_flush"].dt.strftime("%d/%m").fillna("…"),
        }).iloc[::-1]
        st.dataframe(tab, hide_index=True, width="stretch", height=300)
        st.caption("Suelo: agua antes de la lluvia. Emerg.: emergencia total esperada (% de la campaña). "
                   "Flush: mediana y percentil 90 del rezago.")

with st.expander("Cómo funciona y qué precisión tiene"):
    PS = params.summary()
    st.markdown(f"""
**Dos señales.**
1. *Modelo de pulsos*: cada lluvia dispara emergencia con un rezago gamma en tiempo térmico de suelo (mediana
   {PS['mediana']:.0f} °C d sobre {params.Tb:g} °C), una respuesta que sube de febrero a julio ({fmt(PS['julio_febrero'], 1)} veces)
   y hasta {fmt(PS['sequedad'], 1)} veces mayor si la lluvia cae sobre suelo seco. Usa solo la lluvia ya registrada.
2. *Conteos*: la fracción de todo lo contado que salió en el último intervalo. Si la emergencia viene activa, es probable
   que siga; esto capta si la campaña viene adelantada o atrasada, que el clima solo no anticipa.

Cada señal se expresa como percentil de su distribución histórica y la suma se convierte en probabilidad de flush
(semana con ≥ 5 % de la campaña) con una regresión logística.
En el primer conteo con plántulas de la campaña, la señal de conteos vale siempre 100 % y sobreestimaba la
probabilidad (43 % predicho contra 29 % observado); un término de la regresión lo corrige.

**Entrenamiento.** El modelo en uso está ajustado con las 41 campañas disponibles (1977–2015, 2018, 2022 y 2026;
1246 semanas), con más peso a las recientes: el peso de cada campaña se reduce a la mitad cada 20 años hacia atrás,
porque el patrón de emergencia cambió a lo largo de las décadas (más emergencia temprana en otoño). Predecir cada
campaña desde 1990 solo con las anteriores da AUC 0,81 con la ponderación y 0,80 sin ella. La precisión de abajo viene de la validación hecha antes de incorporar las campañas recientes:
el modelo ajustado a 1977–2015 evaluado dejando una campaña afuera y en 2018, 2022 y 2026 como campañas independientes,
medidas con la estación SIGA. Con las 41 campañas, la validación dejando una campaña afuera da el mismo AUC (0,81).

**Precisión:**

| | AUC 1977–2015 | Detección con visitas en 1/3 de las semanas | AUC 2018–2026 |
|---|---|---|---|
| Solo pulsos | 0,76 | ~63–66 % | 0,68 |
| Pulsos + conteos | **0,81** | **~72 %** | **0,76** |

La probabilidad está calibrada: cuando dice 25–30 %, el flush ocurrió en ~25 % de las semanas; cuando dice ~50 %, en ~55 %.

**Niveles.** *Atención* desde 15 %: marca el 48 % de las semanas y cubre el 83 % de los flushes. *Alerta* desde 25 %:
marca el 34 % de las semanas y cubre el 73 %; una de cada 2,3 alertas termina en flush, el doble que una semana cualquiera.

**Clima.** Las temperaturas de SIGA se corrigen por mes a la escala de la estación convencional (NH0343) con la que se
calibró el modelo, usando ERA5 como referencia común.

**Límites.** Un sitio (Bordenave) y suelo sin labranza. Requiere conteos semanales en marcos fijos, con retiro de plántulas.
No anticipa flushes sin lluvia reciente (como el del 9–16 de marzo de 2026) hasta que aparecen en los conteos.
""")
    st.caption(f"Parámetros del modelo de pulsos: {params.source}")

st.sidebar.download_button("📥 Descargar pronósticos (CSV)",
                           fc.assign(prob_pulsos=fc["prob_pulsos"], dF_7d_pct=100 * fc["dF_7d"],
                                     nivel_solo_pulsos=[LEVEL_NAME[level(v)] for v in fc["prob_pulsos"]]).to_csv(index=False),
                           f"avefa_pulsos_{emit:%Y%m%d}.csv")
st.sidebar.caption(f"Clima: {first:%d/%m/%Y} – {last:%d/%m/%Y} · {len(meteo)} días")
