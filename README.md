# AVEFAPREDICTOR2026

Predictores de emergencia de *Avena fatua* para Bordenave (Buenos Aires), en Streamlit.
`meteo_daily.csv` lo usan ambas apps. Se actualiza dos veces por día con GitHub Actions (`actualizar_meteo_bordenave.py`, igual que en LOLIUM_BOR2026), con esta jerarquía de fuentes:

1. **SIGA–INTA Bordenave (estación A872817)**: observado, prioritario y definitivo.
2. **ECMWF IFS histórico**: completa provisionalmente las fechas vencidas que SIGA todavía no publicó (`TipoDato=Provisional`); se reemplaza cuando SIGA las publica.
3. **ECMWF IFS ENS 0,25°**: pronóstico de hoy a 6 días, con la mediana del ensamble (P50) como valor operativo.

La campaña se define con `CAMPANIA_START` y `CAMPANIA_END` en `.github/workflows/actualizar_meteo_bordenave.yml` (hoy 2026-01-01 a 2026-12-31); cambiarlas al empezar 2027. `A872817.xls` es el respaldo local si SIGA no responde.

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

Uso local:

```bash
pip install -r requirements.txt
streamlit run app_pulsos.py
```

Despliegue en Streamlit Community Cloud: *Create app* → este repositorio → rama `main` → archivo principal `app_pulsos.py`.

Límites: predice proporciones de la campaña (no pl/m²); un sitio, suelo sin labranza; pierde habilidad a 14–21 días; no predice la fecha del 50 % mejor que la curva media.

### Validación 2026

`python tools/validar_2026.py` compara el modelo con los conteos de 2026 sin remoción (`data/emergencia_2026_SR.csv`) usando el mismo `meteo_daily.csv` de la app; `tools/figura_validacion_2026.py` genera `docs/validacion_2026.png`.
