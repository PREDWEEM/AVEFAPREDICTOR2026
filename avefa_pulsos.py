"""
avefa_pulsos.py — Modelo de pulsos de lluvia para la emergencia de Avena fatua
=============================================================================

Implementa el modelo de núcleo de respuesta (response-kernel) de:

    Chantre, Carretto, Traversa & Vigna. "Rainfall pulses trigger wild oat
    (Avena fatua) emergence with a soil thermal-time lag: evidence from a
    38-season field record". Weed Research (enviado, 2026).

Ecuación 1 (en forma diaria):

    H(d) = Σ_{j ≤ d} h0(j)  +  Σ_u A(u) · x_u · G(τ_{u,d})
    F(d) = 1 − exp(−H(d))          (fracción del total de la campaña)

    h0(doy)  tasa base: exp(b0 + b1 sin ωd + b2 cos ωd + b3 sin 2ωd + b4 cos 2ωd)
    x_u      ln(1 + ΔW_u),  ΔW_u = min(P_u, W − S_{u−1})
    A(u)     exp(α0 + α1 sin ωd_u + α2 cos ωd_u + δ (1 − S_{u−1}/W))
    G        gamma acumulada del rezago, en tiempo térmico de suelo
             τ = Σ max(Ts − Tb, 0) desde la lluvia, con Tb = 5 °C
    Cada lluvia contribuye solo durante los 60 días siguientes.

Modelo de suelo (Apéndice S1): Ts a 5 cm como media móvil exponencial de la
temperatura media del aire más un desfase estacional; balde de agua de
capacidad W con pérdida k · max(Tmed + 5, 0) · √(amplitud térmica).

El pronóstico a 7 días usa solo la lluvia ya registrada (la lluvia de la
ventana se ignora) y la temperatura de la ventana (pronóstico o climatología).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist

OMEGA = 2 * np.pi / 365.0
RAIN_MIN_MM = 0.25          # día de lluvia: > 0,25 mm
KERNEL_WINDOW_D = 60        # cada lluvia actúa durante 60 días
SEASON_START = (2, 1)       # la emergencia se acumula desde el 1 de febrero


# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
@dataclass
class Params:
    # tasa base h0 (5)
    b0: float
    b1: float
    b2: float
    b3: float
    b4: float
    # respuesta por unidad de lluvia A(u) (3) + sequedad (1)
    alpha0: float
    alpha1: float
    alpha2: float
    delta: float
    # núcleo gamma del rezago (2), en °C d sobre Tb
    k_shape: float
    theta_scale: float
    Tb: float = 5.0
    # modelo de suelo (Apéndice S1)
    ema_alpha: float = 0.426
    ts_a0: float = 3.39
    ts_a1: float = -0.66
    W: float = 75.6
    k_loss: float = 0.0327
    # metadatos
    provisional: bool = True
    source: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Params":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**fields)

    def summary(self) -> dict:
        """Valores derivados con lectura biológica directa."""
        A = lambda d: np.exp(self.alpha1 * np.sin(OMEGA * d) + self.alpha2 * np.cos(OMEGA * d))  # noqa: E731
        return {
            **self.kernel_quantiles(),
            "sequedad": float(np.exp(self.delta)),
            "dia_pico": float((np.arctan2(self.alpha1, self.alpha2) % (2 * np.pi)) / OMEGA),
            "julio_febrero": float(A(196) / A(46)),
        }

    def kernel_quantiles(self) -> dict:
        g = gamma_dist(a=self.k_shape, scale=self.theta_scale)
        return {
            "mediana": float(g.median()),
            "moda": float((self.k_shape - 1) * self.theta_scale),
            "p90": float(g.ppf(0.90)),
        }


# ---------------------------------------------------------------------------
# Datos meteorológicos
# ---------------------------------------------------------------------------
def clean_meteo(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia un CSV con columnas Fecha, TMAX, TMIN, Prec y completa días faltantes."""
    cols = {c.lower(): c for c in df.columns}
    need = ["fecha", "tmax", "tmin", "prec"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}. Se requieren Fecha, TMAX, TMIN, Prec.")
    out = df.rename(columns={cols["fecha"]: "Fecha", cols["tmax"]: "TMAX",
                             cols["tmin"]: "TMIN", cols["prec"]: "Prec"})[["Fecha", "TMAX", "TMIN", "Prec"]]
    out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce", dayfirst=False)
    out = out.dropna(subset=["Fecha"]).sort_values("Fecha").drop_duplicates("Fecha", keep="last")
    for c in ["TMAX", "TMIN", "Prec"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.set_index("Fecha").asfreq("D")
    out[["TMAX", "TMIN"]] = out[["TMAX", "TMIN"]].interpolate(limit_direction="both")
    out["Prec"] = out["Prec"].fillna(0.0).clip(lower=0)
    # si TMIN > TMAX por error de carga, se intercambian
    swap = out["TMIN"] > out["TMAX"]
    out.loc[swap, ["TMAX", "TMIN"]] = out.loc[swap, ["TMIN", "TMAX"]].values
    return out.reset_index()


def climatological_temps(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Temperaturas climatológicas aproximadas de Bordenave, solo para completar
    ventanas de pronóstico que superan los datos disponibles."""
    doy = dates.dayofyear.values
    tmean = 14.8 + 7.8 * np.cos(2 * np.pi * (doy - 20) / 365.0)
    dtr = 13.5 + 2.5 * np.cos(2 * np.pi * (doy - 20) / 365.0)
    return pd.DataFrame({"Fecha": dates, "TMAX": tmean + dtr / 2, "TMIN": tmean - dtr / 2, "Prec": 0.0})


def extend_with_climatology(df: pd.DataFrame, until: pd.Timestamp) -> pd.DataFrame:
    last = df["Fecha"].max()
    if until <= last:
        return df
    extra = climatological_temps(pd.date_range(last + pd.Timedelta(days=1), until, freq="D"))
    extra["climatologico"] = True
    out = pd.concat([df.assign(climatologico=df.get("climatologico", False)), extra], ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Modelo de suelo
# ---------------------------------------------------------------------------
def soil_model(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Agrega Tmed, Ts (5 cm), S (agua del balde al final del día) y S_prev."""
    tmax = df["TMAX"].to_numpy(float)
    tmin = df["TMIN"].to_numpy(float)
    rain = df["Prec"].to_numpy(float)
    doy = pd.DatetimeIndex(df["Fecha"]).dayofyear.to_numpy()
    tmean = (tmax + tmin) / 2
    dtr = np.clip(tmax - tmin, 0, None)

    n = len(df)
    ema = np.empty(n)
    S = np.empty(n)
    s_prev = np.empty(n)
    e, s = tmean[0], p.W / 2  # arranque: balde a media capacidad
    for i in range(n):
        e = p.ema_alpha * tmean[i] + (1 - p.ema_alpha) * e if i else tmean[0]
        ema[i] = e
        s_prev[i] = s
        s = min(s + rain[i], p.W)
        s = max(s - p.k_loss * max(tmean[i] + 5, 0) * np.sqrt(dtr[i]), 0.0)
        S[i] = s
    ts = ema + p.ts_a0 + p.ts_a1 * np.cos(2 * np.pi * (doy - 196) / 365.0)

    out = df.copy()
    out["Tmed"] = tmean
    out["Ts"] = ts
    out["S"] = S
    out["S_prev"] = s_prev
    out["TT"] = np.clip(ts - p.Tb, 0, None)
    out["C_TT"] = np.cumsum(out["TT"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Núcleo del modelo
# ---------------------------------------------------------------------------
def baseline_rate(doy: np.ndarray, p: Params) -> np.ndarray:
    w = OMEGA * doy
    return np.exp(p.b0 + p.b1 * np.sin(w) + p.b2 * np.cos(w) + p.b3 * np.sin(2 * w) + p.b4 * np.cos(2 * w))


def response_A(doy: np.ndarray, s_prev: np.ndarray, p: Params) -> np.ndarray:
    w = OMEGA * doy
    return np.exp(p.alpha0 + p.alpha1 * np.sin(w) + p.alpha2 * np.cos(w) + p.delta * (1 - s_prev / p.W))


def seasonal_response_curve(p: Params) -> pd.DataFrame:
    doy = np.arange(1, 366)
    a = np.exp(p.alpha1 * np.sin(OMEGA * doy) + p.alpha2 * np.cos(OMEGA * doy))
    return pd.DataFrame({"doy": doy, "respuesta_rel": a / a.max()})


@dataclass
class Simulation:
    days: pd.DataFrame          # serie diaria con suelo, H, F
    events: pd.DataFrame        # días de lluvia con su respuesta esperada
    P: np.ndarray               # contribución acumulada de cada lluvia (n_eventos × n_días)
    B: np.ndarray               # hazard base acumulado
    rain_idx: np.ndarray        # índice (en days) de cada lluvia
    start_idx: int              # índice del 1 de febrero


def simulate(df_meteo: pd.DataFrame, p: Params, rain_cutoff: pd.Timestamp | None = None,
             season_year: int | None = None) -> Simulation:
    """Simula la campaña desde el 1 de febrero con la lluvia registrada hasta rain_cutoff."""
    d = soil_model(df_meteo, p)
    dates = pd.DatetimeIndex(d["Fecha"])
    if season_year is None:
        season_year = int(dates[-1].year if dates[-1].month >= 2 else dates[-1].year - 1)
    start = pd.Timestamp(season_year, *SEASON_START)
    if start < dates[0]:
        start = dates[0]
    start_idx = int(np.searchsorted(dates.values, start.to_datetime64()))
    n = len(d)
    doy = dates.dayofyear.to_numpy()
    C = d["C_TT"].to_numpy()

    # hazard base acumulado desde el 1 de febrero
    h0 = baseline_rate(doy, p)
    h0[:start_idx] = 0.0
    B = np.cumsum(h0)

    rain = d["Prec"].to_numpy().copy()
    if rain_cutoff is not None:
        rain[dates > rain_cutoff] = 0.0
    rain_mask = rain > RAIN_MIN_MM
    # solo lluvias que todavía actúan desde el 1 de febrero
    first_u = start_idx - KERNEL_WINDOW_D
    rain_mask[: max(first_u, 0)] = False
    rain_idx = np.where(rain_mask)[0]

    s_prev = d["S_prev"].to_numpy()
    dW = np.minimum(rain, p.W - s_prev)
    x = np.log1p(np.clip(dW, 0, None))
    A = response_A(doy, s_prev, p)

    g = gamma_dist(a=p.k_shape, scale=p.theta_scale)
    P = np.zeros((len(rain_idx), n))
    day_idx = np.arange(n)
    for r, u in enumerate(rain_idx):
        end = min(u + KERNEL_WINDOW_D, n - 1)
        tau = np.clip(C[np.minimum(day_idx, end)] - C[u], 0, None)
        contrib = A[u] * x[u] * g.cdf(tau)
        contrib[day_idx < u] = 0.0
        # lo acumulado antes del 1 de febrero no cuenta
        if u < start_idx:
            contrib = np.where(day_idx >= start_idx, contrib - contrib[start_idx - 1], 0.0)
        P[r] = contrib

    H = B + P.sum(axis=0)
    F = 1 - np.exp(-H)
    d["h0_acum"] = B
    d["H"] = H
    d["F"] = F
    d["en_campaña"] = day_idx >= start_idx

    ev = d.iloc[rain_idx][["Fecha", "Prec", "S_prev", "Ts"]].copy()
    ev["agua_agregada_mm"] = dW[rain_idx]
    ev["respuesta_A"] = A[rain_idx]
    ev["multiplicador_sequedad"] = np.exp(p.delta * (1 - s_prev[rain_idx] / p.W))
    ev["hazard_total"] = A[rain_idx] * x[rain_idx]
    q = p.kernel_quantiles()
    ev["fecha_mediana_flush"] = [_date_at_tt(dates, C, u, q["mediana"]) for u in rain_idx]
    ev["fecha_p90_flush"] = [_date_at_tt(dates, C, u, q["p90"]) for u in rain_idx]
    return Simulation(days=d, events=ev.reset_index(drop=True), P=P, B=B, rain_idx=rain_idx, start_idx=start_idx)


def _date_at_tt(dates: pd.DatetimeIndex, C: np.ndarray, u: int, target: float):
    need = C[u] + target
    j = int(np.searchsorted(C, need))
    return dates[j] if j < len(dates) else pd.NaT


# ---------------------------------------------------------------------------
# Pronósticos
# ---------------------------------------------------------------------------
def rolling_forecast(sim: Simulation, horizon: int = 7, F0_override: dict | None = None) -> pd.DataFrame:
    """Pronóstico emitido cada día t para (t, t+horizon], usando solo lluvias con u ≤ t.

    F0_override: {fecha: fracción ya emergida} para el modo "con conteos".
    """
    d = sim.days
    n = len(d)
    dates = pd.DatetimeIndex(d["Fecha"])
    rows = []
    cum_P_by_t = np.cumsum(sim.P, axis=0) if len(sim.P) else np.zeros((0, n))
    for t in range(sim.start_idx, n - horizon):
        # lluvias con u ≤ t
        k = int(np.searchsorted(sim.rain_idx, t, side="right"))
        Ht_t = sim.B[t] + (cum_P_by_t[k - 1, t] if k else 0.0)
        Ht_e = sim.B[t + horizon] + (cum_P_by_t[k - 1, t + horizon] if k else 0.0)
        dH = Ht_e - Ht_t
        F_t = 1 - np.exp(-Ht_t)
        if F0_override:
            key = dates[t].normalize()
            if key in F0_override:
                F_t = F0_override[key]
        dF = (1 - F_t) * (1 - np.exp(-dH))
        dF_base = (1 - F_t) * (1 - np.exp(-(sim.B[t + horizon] - sim.B[t])))
        rows.append((dates[t], F_t, dF, dF_base))
    return pd.DataFrame(rows, columns=["Fecha", "F_emitido", "dF_7d", "dF_7d_base"])


CLIM_ANCHORS = {"02-01": 0.0, "03-15": 0.07, "05-15": 0.38, "07-15": 0.67, "09-15": 0.90, "10-31": 0.975}


def climatological_curve(dates: pd.DatetimeIndex) -> np.ndarray:
    """Curva media de emergencia acumulada de Bordenave (1977–2015), interpolada."""
    from scipy.interpolate import PchipInterpolator
    doy_a = [pd.Timestamp(f"2001-{k}").dayofyear for k in CLIM_ANCHORS]
    f = PchipInterpolator(doy_a, list(CLIM_ANCHORS.values()), extrapolate=False)
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    out = f(np.clip(doy, doy_a[0], doy_a[-1]))
    out[doy < doy_a[0]] = 0.0
    return out


def classify_alert(dF: float, thr_low: float = 0.02, thr_high: float = 0.03) -> str:
    if dF >= thr_high:
        return "alta"
    if dF >= thr_low:
        return "moderada"
    return "baja"


def single_rain_response(p: Params, doy: int, mm: float, s_prev_frac: float, remaining: float) -> float:
    """Emergencia total esperada (fracción de la campaña) para una lluvia aislada."""
    s_prev = s_prev_frac * p.W
    x = np.log1p(min(mm, p.W - s_prev))
    A = response_A(np.array([doy]), np.array([s_prev]), p)[0]
    return remaining * (1 - np.exp(-A * x))
