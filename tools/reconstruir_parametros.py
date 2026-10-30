"""
Reconstruye un vector de parámetros PROVISORIO para avefa_pulsos.py a partir de
los valores derivados publicados en el manuscrito (Weed Research, enviado 2026).

El manuscrito no reporta los 11 parámetros ajustados, sino cantidades derivadas.
Este script las invierte:

  Núcleo gamma      moda 111 °C d y mediana 139 °C d  → forma k, escala θ
                    (el percentil 90 publicado, 260 °C d, queda como control)
  Respuesta A(u)    pico en el día 194 y cociente mediados de julio / mediados
                    de febrero = 20 (estimación puntual, Tabla S2) → α1, α2
  Sequedad          multiplicador 4,2 (suelo vacío vs lleno) → δ = ln 4,2
  Escala α0         Fig. 4: 20 mm a fines de junio sobre suelo seco ≈ 14 % de
                    la campaña, con la proporción media que falta emerger
  Tasa base h0      5 coeficientes calibrados para que la curva media simulada
                    con un conjunto de campañas sintéticas reproduzca la curva
                    climatológica (7 % a mediados de marzo, 38 % mayo, 67 % julio,
                    90 % septiembre; Bordenave 1977–2015)

El resultado se guarda en pulsos_params_reconstruidos.json, solo como
referencia. La app usa pulsos_params.json, que tiene el vector ajustado.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize
from scipy.stats import gamma as gamma_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import avefa_pulsos as ap  # noqa: E402

OUT = ROOT / "pulsos_params_reconstruidos.json"

# --- 1. núcleo gamma -------------------------------------------------------
MODE, MEDIAN, P90_PUB = 111.0, 139.0, 260.0


def median_given_k(k):
    theta = MODE / (k - 1)
    return gamma_dist(a=k, scale=theta).median() - MEDIAN


k_shape = brentq(median_given_k, 1.05, 50)
theta = MODE / (k_shape - 1)
p90 = gamma_dist(a=k_shape, scale=theta).ppf(0.9)

# --- 2. respuesta estacional -------------------------------------------------
PEAK_DOY, RATIO, DOY_JUL, DOY_FEB = 194, 20.0, 196, 46
w = 2 * np.pi / 365
amp = np.log(RATIO) / (np.cos(w * (DOY_JUL - PEAK_DOY)) - np.cos(w * (DOY_FEB - PEAK_DOY)))
# amp·cos(w(d − peak)) = α1 sin(wd) + α2 cos(wd)
alpha1 = amp * np.sin(w * PEAK_DOY)
alpha2 = amp * np.cos(w * PEAK_DOY)

# --- 3. sequedad -------------------------------------------------------------
delta = np.log(4.2)

# --- curva climatológica (anclas) -------------------------------------------
anchors = pd.DataFrame({
    "fecha": ["02-01", "03-15", "05-15", "07-15", "09-15", "10-31"],
    "F": [0.0, 0.07, 0.38, 0.67, 0.90, 0.975],
})
anchors["doy"] = pd.to_datetime("2001-" + anchors["fecha"]).dt.dayofyear
clim = PchipInterpolator(anchors["doy"], anchors["F"])

# --- 4. escala α0 desde la Fig. 4 ---------------------------------------------
DOY_FIG4, MM, TARGET = 177, 20.0, 0.14
remaining = 1 - float(clim(DOY_FIG4))
H_needed = -np.log(1 - TARGET / remaining)
x = np.log1p(MM)
seasonal = amp * np.cos(w * (DOY_FIG4 - PEAK_DOY))
alpha0 = np.log(H_needed / x) - seasonal - delta  # suelo vacío: término de sequedad = δ

# --- 5. tasa base h0 con campañas sintéticas ----------------------------------
rng = np.random.default_rng(1977)
N_SEASONS = 80
# frecuencia relativa de lluvias por mes (región semiárida del SO bonaerense)
month_w = {1: 70, 2: 70, 3: 90, 4: 70, 5: 45, 6: 30, 7: 30, 8: 30, 9: 55, 10: 75, 11: 70, 12: 70}
EVENTS_FEB_OCT = 31.6
SIGMA = np.log(20 / 7) / 0.7063  # mediana 7 mm, 24 % ≥ 20 mm


def synthetic_season(year=2001):
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    base = ap.climatological_temps(dates)
    noise = np.zeros(len(dates))
    for i in range(1, len(dates)):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 2.0)
    base["TMAX"] += noise
    base["TMIN"] += noise
    feb_oct = sum(month_w[m] * pd.Period(f"{year}-{m:02d}").days_in_month for m in range(2, 11))
    per_day = np.array([month_w[d.month] for d in dates]) * EVENTS_FEB_OCT / feb_oct
    wet = rng.random(len(dates)) < per_day
    base["Prec"] = np.where(wet, np.exp(np.log(7) + SIGMA * rng.standard_normal(len(dates))), 0.0)
    return base


params = dict(b0=-6.0, b1=0, b2=0, b3=0, b4=0, alpha0=alpha0, alpha1=alpha1, alpha2=alpha2,
              delta=delta, k_shape=k_shape, theta_scale=theta)
p_tmp = ap.Params(**params)

sims = [ap.simulate(synthetic_season(), p_tmp, season_year=2001) for _ in range(N_SEASONS)]
P_sum = np.array([s.P.sum(axis=0) for s in sims])
doy = pd.DatetimeIndex(sims[0].days["Fecha"]).dayofyear.to_numpy()
start = sims[0].start_idx
anchor_idx = anchors["doy"].to_numpy()[1:] - 1  # índices (año no bisiesto)
target = anchors["F"].to_numpy()[1:]


def mean_curve(b):
    pb = ap.Params(**{**params, "b0": b[0], "b1": b[1], "b2": b[2], "b3": b[3], "b4": b[4]})
    h0 = ap.baseline_rate(doy, pb)
    h0[:start] = 0
    B = np.cumsum(h0)
    F = 1 - np.exp(-(B[None, :] + P_sum))
    return F.mean(axis=0)


def loss(b):
    F = mean_curve(b)
    return np.sum((F[anchor_idx] - target) ** 2) * 1e4 + 0.05 * (b[3] ** 2 + b[4] ** 2)


best = None
for b0 in (-6.0, -5.0, -4.0):
    r = minimize(loss, x0=[b0, 0.5, -0.5, 0, 0], method="Nelder-Mead",
                 options={"maxiter": 6000, "xatol": 1e-5, "fatol": 1e-8})
    if best is None or r.fun < best.fun:
        best = r
b = best.x
Fm = mean_curve(b)

pulse_share = []
for i in range(N_SEASONS):
    pb = ap.Params(**{**params, "b0": b[0], "b1": b[1], "b2": b[2], "b3": b[3], "b4": b[4]})
    h0 = ap.baseline_rate(doy, pb)
    h0[:start] = 0
    pulse_share.append(P_sum[i, -1] / (P_sum[i, -1] + h0.sum()))

result = {
    **{k: float(v) for k, v in params.items()},
    "b0": float(b[0]), "b1": float(b[1]), "b2": float(b[2]), "b3": float(b[3]), "b4": float(b[4]),
    "Tb": 5.0, "ema_alpha": 0.426, "ts_a0": 3.39, "ts_a1": -0.66, "W": 75.6, "k_loss": 0.0327,
    "provisional": True,
    "source": ("Reconstruido de valores derivados del manuscrito (rezago 111/139 °C d, pico día 194, "
               "cociente 20, sequedad 4,2, Fig. 4) y curva climatológica de Bordenave. "
               "Reemplazar por el ajuste real de 1977–2015."),
    "_diagnostico": {
        "kernel_p90_reconstruido": float(p90),
        "kernel_p90_publicado": P90_PUB,
        "curva_media_en_anclas": {a: round(float(Fm[i]), 3) for a, i in zip(anchors["fecha"][1:], anchor_idx)},
        "curva_objetivo": {a: f for a, f in zip(anchors["fecha"][1:], target)},
        "fraccion_hazard_por_pulsos": round(float(np.mean(pulse_share)), 3),
        "n_campañas_sinteticas": N_SEASONS,
    },
}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
