# AVEFAPREDICTOR2026

Predictores de emergencia de *Avena fatua* para Bordenave (Buenos Aires), en Streamlit.
`meteo_daily.csv` se actualiza todos los días con GitHub Actions (`update_meteo.py`, MeteoBahía) y lo usan ambas apps.

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
- `pulsos_params.json`: parámetros. **Hoy son provisorios** (`"provisional": true`), reconstruidos con `tools/reconstruir_parametros.py` a partir de los valores derivados del manuscrito. Reemplazarlos por el vector ajustado a 1977–2015 y poner `"provisional": false` (desaparece el aviso de la app).
- `app_pulsos.py`: interfaz. Admite `?fecha=AAAA-MM-DD` en la URL y la carga de un CSV propio (`Fecha, TMAX, TMIN, Prec`).

Uso local:

```bash
pip install -r requirements.txt
streamlit run app_pulsos.py
```

Despliegue en Streamlit Community Cloud: *Create app* → este repositorio → rama `main` → archivo principal `app_pulsos.py`.

Límites: predice proporciones de la campaña (no pl/m²); un sitio, suelo sin labranza; pierde habilidad a 14–21 días; no predice la fecha del 50 % mejor que la curva media.
