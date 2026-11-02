# AVEFAPREDICTOR2026

Predictores de emergencia de *Avena fatua* para Bordenave (Buenos Aires), en Streamlit.
`meteo_daily.csv` lo usan ambas apps. Se actualiza dos veces por día con GitHub Actions (`actualizar_meteo_bordenave.py`, igual que en LOLIUM_BOR2026), con esta jerarquía de fuentes:

1. **SIGA–INTA Bordenave (estación A872817)**: observado, prioritario y definitivo.
2. **ECMWF IFS histórico**: completa provisionalmente las fechas vencidas que SIGA todavía no publicó (`TipoDato=Provisional`); se reemplaza cuando SIGA las publica.
3. **ECMWF IFS ENS 0,25°**: pronóstico de hoy a 6 días, con la mediana del ensamble (P50) como valor operativo.

La campaña es automática: del 1 de enero al 31 de diciembre del año en curso (hora de Argentina). El **1 de enero se reinicia sola la carga** con la campaña nueva (la de 2027 empieza el 1/01/2027): en la primera actualización del año la serie de la campaña anterior se guarda en `data/campanias/<año>/clima_bordenave_<año>.csv` (no se sobrescribe si ya existe), la caché SIGA se limita a la campaña vigente y, durante las primeras tres semanas, se admite una serie sin observaciones SIGA del año (SIGA publica con demora), cubierta con el puente provisional y el pronóstico ECMWF. Las variables `CAMPANIA_START` y `CAMPANIA_END` (AAAA-MM-DD) permiten fijar otras fechas. `A872817.xls` es el respaldo local si SIGA no responde. Antes del 1 de febrero la app AVEFA Pulsos avisa que la campaña todavía no empezó.

| App | Archivo principal | Modelo |
|---|---|---|
| PREDWEEM v8.5 | `app_predweem_v8.py` | Red neuronal (Chantre et al., 2018) |
| AVEFA Pulsos | `app_pulsos.py` | Pulsos de lluvia con rezago en tiempo térmico (Chantre et al., Weed Research, enviado 2026) |

## AVEFA Pulsos

Alerta semanal de flushes: cuánto de la emergencia de la campaña se espera en los próximos 7 días, usando solo la lluvia ya registrada.

- Cada lluvia dispara emergencia con un rezago gamma en tiempo térmico de suelo a 5 cm sobre 5 °C (mediana 139 °C d: ~7 días en febrero, ~28 en julio).
- La respuesta por unidad de lluvia sube ~20 veces de febrero a principios de julio, y es hasta 4,2 veces mayor si la lluvia cae sobre suelo seco.
- Sin umbral mínimo de lluvia; la respuesta crece con el logaritmo del agua que la lluvia agrega al suelo.

Archivos:

- `avefa_pulsos.py`: modelo de suelo (Apéndice S1), Ecuación 1 en forma diaria, pronóstico rodante a 7 días.
- `pulsos_params.json`: vector θ de 11 parámetros ajustado por mínimos cuadrados no lineales a los intervalos de 1977–2015 con una reimplementación en Python (no es el vector del código original del manuscrito). La forma y la escala del núcleo gamma se guardan ya exponenciadas (`k_shape = exp θ₁₀`, `theta_scale = exp θ₁₁`). Si `"provisional": true`, la app muestra un aviso.
- `pulsos_params_reconstruidos.json`: vector alternativo reconstruido con `tools/reconstruir_parametros.py` a partir de los valores derivados publicados; queda como referencia.
- `app_pulsos.py`: interfaz. Admite `?fecha=AAAA-MM-DD` en la URL y la carga de un CSV propio (`Fecha, TMAX, TMIN, Prec`).
  Al final de la página, **Descargar resultados en Excel** reúne lo que muestra la app para la fecha de emisión: resumen
  (nivel, probabilidad, muestreo), alerta diaria del modelo de pulsos, alerta con conteos, conteos, emergencia acumulada
  (simulada, curva media y estimada con conteos y su rango), lluvias de los últimos 60 días y clima.

### Alerta combinada (pulsos + conteos)

Con conteos semanales de plántulas (marcos fijos, con retiro), la app combina dos señales: el pronóstico del modelo de pulsos
para los próximos 7 días y la fracción de lo contado en la campaña que emergió en el último intervalo. Cada una se expresa
como percentil respecto de 1977–2015 y la suma se convierte en probabilidad de flush (≥ 5 % de la campaña en la semana)
con una regresión logística (`alerta_params.json`).

| Validación | AUC solo pulsos | AUC pulsos + conteos | Detección con visitas en 1/3 de las semanas |
|---|---|---|---|
| 1977–2015, dejando una campaña afuera (1163 semanas) | 0,76 | 0,81 | ~63 % → ~72 % |
| 2018, 2022 y 2026, estación SIGA (83 semanas) | 0,68 | 0,76 | ~50 % → ~71 % |

La regresión incluye un término para el primer conteo con plántulas de la campaña, donde la fracción del último
intervalo es siempre 1 y sobreestimaba la probabilidad (43 % predicho contra 29 % observado; con el término, 31 %).
La app muestra dos niveles fijos: **Atención** desde 15 % de probabilidad (48 % de las semanas marcadas, 83 % de los
flushes cubiertos) y **Alerta** desde 25 % (34 % de las semanas, 73 % de los flushes), según la validación con las 41
campañas dejando una afuera.
Con conteos, la app estima además qué parte de la campaña ya emergió y cuánto falta, con un rango del 80 % calibrado en 41 campañas (`remanente_params.json`; cobertura fuera de muestra 78 %). La app indica cuándo contar: cada semana mientras haya Alerta y cada dos semanas el resto (simulado en 41 campañas: AUC 0,794 contra 0,808 con conteo semanal fijo, con un tercio menos de conteos). Un conteo vale hasta 14 días; sin conteos recientes, la app usa solo el modelo de pulsos. Los conteos se cargan en la tabla de la barra
lateral o desde un CSV (`Fecha, Marco 1, Marco 2, …`); no se guardan en el repositorio.

Modelo en uso (`pulsos_params.json` y `alerta_params.json`): ajustado con las 41 campañas disponibles (1977–2015 con
NH0343 corregida —meses anómalos de 2001–2005 reemplazados por ERA5 + sesgo de la estación— y 2018, 2022 y 2026 con
SIGA corregida; 1246 semanas), con mínimos cuadrados ponderados: cada campaña pesa 0,5^((2026 − año)/20), porque
el patrón de emergencia cambió a lo largo de las décadas. En validación hacia adelante (cada campaña desde 1990
predicha solo con las anteriores) la alerta pasa de AUC 0,800 a 0,810 (IC95 del cambio +0,003 a +0,018). El ajuste
sin ponderar está en `pulsos_params_41_sin_ponderar.json` y `alerta_params_41_sin_ponderar.json`. La tabla de precisión corresponde a la validación previa con el modelo de 1977–2015,
guardado en `pulsos_params_1977_2015.json`; el vector de la reimplementación del manuscrito está en
`pulsos_params_usuario.json`. La
temperatura de SIGA se corrige por mes a la escala de NH0343 (`correccion_temperatura`).

Uso local:

```bash
pip install -r requirements.txt
streamlit run app_pulsos.py
```

Despliegue en Streamlit Community Cloud: *Create app* → este repositorio → rama `main` → archivo principal `app_pulsos.py`.

Límites: predice proporciones de la campaña (no pl/m²); un sitio, suelo sin labranza; pierde habilidad a 14–21 días; no predice la fecha del 50 % mejor que la curva media.

### Validación 2026

`python tools/validar_2026.py` compara el modelo con los conteos de 2026 sin remoción (`data/emergencia_2026_SR.csv`) usando el clima de la app (`meteo_daily.csv` mientras dure la campaña 2026; desde el 1/1/2027, la copia `data/campanias/2026/clima_bordenave_2026.csv`); `tools/figura_validacion_2026.py` genera `docs/validacion_2026.png`.

## Apps siempre activas

Streamlit Community Cloud suspende una app después de 12 horas sin visitas. El workflow
[`mantener_activas.yml`](.github/workflows/mantener_activas.yml) abre **AVEFA Pulsos** y **PREDWEEM v8.5** con Chromium
cada cuatro horas (a los 17 minutos de las 00, 04, 08, 12, 16 y 20 UTC), también al modificarse y a mano desde
**Actions → Mantener activas las apps Streamlit → Run workflow**. El script (`scripts/mantener_activas.py`):

- pulsa el botón para despertar la app si está suspendida;
- espera hasta cinco minutos a que la interfaz esté cargada de verdad (título y barra lateral, incluso dentro del iframe
  de Streamlit); una respuesta HTTP 200 sola no cuenta;
- falla si alguna app no carga o muestra una excepción, y GitHub lo avisa según la configuración de notificaciones.

Con seis visitas por día queda margen aunque GitHub demore las ejecuciones programadas. GitHub desactiva los workflows
programados después de 60 días sin actividad en el repositorio; aquí no ocurre porque la actualización del clima hace
commits dos veces por día. Esto reduce el riesgo de suspensión, pero no garantiza disponibilidad continua.
