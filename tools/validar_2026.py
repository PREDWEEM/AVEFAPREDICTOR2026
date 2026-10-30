"""
Validación del modelo de pulsos con la campaña 2026 (sin remoción, Bordenave).

Datos:  data/emergencia_2026_SR.csv  (conteos por marco; media por m²)
Clima:  meteo_daily.csv              (SIGA–INTA Bordenave + ECMWF, el mismo que usa la app)

Se evalúa, como en la Sección 2.5 del manuscrito:
  1. Curva acumulada simulada solo con clima desde el 1 de febrero (RMSE y
     fecha del 50 %), contra la curva climatológica del modelo solo de base.
  2. Pronóstico para el intervalo siguiente a cada conteo con la lluvia
     registrada hasta la fecha del conteo, con F0 de la simulación (solo
     clima) o de los conteos. Skill = 1 − ECM/ECM_ref; AUC para intervalos
     con ≥ 5 % de la campaña; r de Pearson.
  3. Regla de monitoreo: visitar si el pronóstico solo con clima ≥ 3 % o 2 %.

Uso:  python tools/validar_2026.py [--csv salida.csv]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import avefa_pulsos as ap  # noqa: E402

# línea de base sola, ajustada a 1977–2015 (referencia del manuscrito)
BASE_ONLY = dict(b0=-5.546, b1=-0.399, b2=-1.875, b3=0.493, b4=-0.912)
FLUSH = 0.05


def load_counts(path=ROOT / "data" / "emergencia_2026_SR.csv"):
    c = pd.read_csv(path, parse_dates=["Fecha"])
    c["media"] = c[["m1", "m2", "m3"]].mean(axis=1, skipna=True)
    start = c["Fecha"].iloc[0]
    c = c.dropna(subset=["media"]).reset_index(drop=True)
    c["inicio"] = [start] + list(c["Fecha"].iloc[:-1])
    total = c["media"].sum()
    c["dF_obs"] = c["media"] / total
    c["F_obs"] = c["dF_obs"].cumsum()
    c["F0_obs"] = c["F_obs"] - c["dF_obs"]
    c["dias"] = (c["Fecha"] - c["inicio"]).dt.days
    return c, total


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return greater / (len(pos) * len(neg))


def date_at(dates, F, level=0.5):
    F = np.asarray(F)
    j = int(np.argmax(F >= level))
    if F[j] < level:
        return pd.NaT
    if j == 0:
        return dates[0]
    frac = (level - F[j - 1]) / (F[j] - F[j - 1])
    return dates[j - 1] + (dates[j] - dates[j - 1]) * frac


def evaluate(params: ap.Params, counts: pd.DataFrame, meteo: pd.DataFrame, label: str):
    sim = ap.simulate(meteo, params, rain_cutoff=None, season_year=2026)
    d = sim.days
    dates = pd.DatetimeIndex(d["Fecha"])
    idx = {t: i for i, t in enumerate(dates)}
    doy = dates.dayofyear.to_numpy()

    pb = ap.Params(**{**params.__dict__, **BASE_ONLY})
    h_base = ap.baseline_rate(doy, pb)
    h_base[: sim.start_idx] = 0
    Bb = np.cumsum(h_base)
    F_clim = 1 - np.exp(-Bb)
    cumP = np.cumsum(sim.P, axis=0)

    rows = []
    for _, r in counts.iterrows():
        s, e = idx[r["inicio"]], idx[r["Fecha"]]
        k = int(np.searchsorted(sim.rain_idx, s, side="right"))
        pulses = (cumP[k - 1, e] - cumP[k - 1, s]) if k else 0.0
        dH = sim.B[e] - sim.B[s] + pulses
        dH_full = d["H"].iloc[e] - d["H"].iloc[s]
        F0_sim = float(d["F"].iloc[s])
        F0_obs = r["F0_obs"]
        rows.append({
            "inicio": r["inicio"], "fin": r["Fecha"], "dias": r["dias"], "dF_obs": r["dF_obs"],
            "F_obs": r["F_obs"], "F_sim": float(d["F"].iloc[e]), "F_clim": F_clim[e],
            "pred_clima": (1 - F0_sim) * (1 - np.exp(-dH)),
            "pred_clima_lluvia_futura": (1 - F0_sim) * (1 - np.exp(-dH_full)),
            "ref_clima": F_clim[e] - F_clim[s],
            "pred_conteos": (1 - F0_obs) * (1 - np.exp(-dH)),
            "pred_conteos_lluvia_futura": (1 - F0_obs) * (1 - np.exp(-dH_full)),
            "ref_conteos": (1 - F0_obs) * (1 - np.exp(-(Bb[e] - Bb[s]))),
        })
    v = pd.DataFrame(rows)
    y = v["dF_obs"].to_numpy()
    flush = y >= FLUSH

    def skill(p, ref):
        return 1 - np.mean((v[p] - y) ** 2) / np.mean((v[ref] - y) ** 2)

    def corr(p):
        return float(np.corrcoef(v[p], y)[0, 1])

    obs_dates = pd.DatetimeIndex(v["fin"])
    d50_obs = date_at(obs_dates, v["F_obs"])
    d50_sim = date_at(dates, d["F"])
    d50_clim = date_at(dates, F_clim)
    window = (dates >= obs_dates[0]) & (dates <= obs_dates[-1])

    res = {
        "vector": label,
        "intervalos": len(v), "flushes": int(flush.sum()),
        "curva_RMSE_modelo_pp": 100 * float(np.sqrt(np.mean((v["F_sim"] - v["F_obs"]) ** 2))),
        "curva_RMSE_clim_pp": 100 * float(np.sqrt(np.mean((v["F_clim"] - v["F_obs"]) ** 2))),
        "D50_obs": d50_obs.strftime("%d/%m"),
        "D50_modelo": d50_sim.strftime("%d/%m"), "D50_err_modelo_d": (d50_sim - d50_obs).days,
        "D50_clim": d50_clim.strftime("%d/%m"), "D50_err_clim_d": (d50_clim - d50_obs).days,
        "skill_clima": skill("pred_clima", "ref_clima"),
        "skill_clima_lluvia_futura": skill("pred_clima_lluvia_futura", "ref_clima"),
        "AUC_ref_clima": auc(v["ref_clima"].to_numpy(), flush),
        "AUC_clima": auc(v["pred_clima"].to_numpy(), flush),
        "r_ref_clima": corr("ref_clima"), "r_clima": corr("pred_clima"),
        "skill_conteos": skill("pred_conteos", "ref_conteos"),
        "skill_conteos_lluvia_futura": skill("pred_conteos_lluvia_futura", "ref_conteos"),
        "AUC_ref_conteos": auc(v["ref_conteos"].to_numpy(), flush),
        "AUC_conteos": auc(v["pred_conteos"].to_numpy(), flush),
        "r_ref_conteos": corr("ref_conteos"), "r_conteos": corr("pred_conteos"),
    }
    for thr in (0.03, 0.02):
        visit = v["pred_clima"].to_numpy() >= thr
        res[f"regla_{int(thr*100)}pct_visitas"] = f"{visit.sum()}/{len(v)}"
        res[f"regla_{int(thr*100)}pct_flushes"] = f"{(visit & flush).sum()}/{flush.sum()}"
    return res, v, sim, F_clim, window


def main():
    counts, total = load_counts()
    meteo = ap.clean_meteo(pd.read_csv(ROOT / "meteo_daily.csv"))
    out = []
    for label, f in [("ajustado", "pulsos_params.json"), ("reconstruido", "pulsos_params_reconstruidos.json")]:
        res, v, *_ = evaluate(ap.Params.load(ROOT / f), counts, meteo, label)
        out.append(res)
        if label == "ajustado":
            v_main = v
    print(f"Total de la campaña: {total:.0f} pl/m² · {len(counts)} intervalos")
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    if "--csv" in sys.argv:
        v_main.to_csv(sys.argv[sys.argv.index("--csv") + 1], index=False)


if __name__ == "__main__":
    main()
