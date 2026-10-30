"""
Escenario: ¿una lluvia no medida por la estación explica el flush del 9–16 de marzo de 2026?

Agrega a la serie SIGA una lluvia de monto y fecha variables (o las lluvias que
registró MeteoBahía) y recalcula el pronóstico del intervalo 9–16 de marzo y las
métricas de validación de la campaña.

Uso:  python tools/escenario_lluvia_no_medida.py [figura.png]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import avefa_pulsos as ap  # noqa: E402
from validar_2026 import evaluate, load_counts  # noqa: E402

FLUSH_INI, FLUSH_FIN = pd.Timestamp("2026-03-09"), pd.Timestamp("2026-03-16")
FECHAS = pd.to_datetime(["2026-02-24", "2026-02-28", "2026-03-02", "2026-03-04", "2026-03-07"])
MONTOS = [5, 10, 20, 30, 50, 75]

counts, total = load_counts()
base = ap.clean_meteo(pd.read_csv(ROOT / "meteo_daily.csv"))
params = ap.Params.load(ROOT / "pulsos_params.json")
obs = float(counts.loc[counts["Fecha"] == FLUSH_FIN, "dF_obs"].iloc[0])


def with_rain(extra: dict) -> pd.DataFrame:
    m = base.copy()
    for f, mm in extra.items():
        m.loc[m["Fecha"] == pd.Timestamp(f), "Prec"] += mm
    return m


def run(extra: dict, label: str):
    res, v, *_ = evaluate(params, counts, with_rain(extra), label)
    row = v.loc[v["fin"] == FLUSH_FIN].iloc[0]
    return res, float(row["pred_clima"]), float(row["pred_conteos"])


rows = []
for f in FECHAS:
    for mm in MONTOS:
        _, pc, pk = run({f: mm}, "grilla")
        rows.append({"fecha": f, "mm": mm, "pred_clima": pc, "pred_conteos": pk})
grid = pd.DataFrame(rows)

escenarios = {
    "Estación (sin lluvia agregada)": {},
    "Lluvias de MeteoBahía (28/2: 18 mm; 2/3: 27 mm)": {"2026-02-28": 18.2, "2026-03-02": 26.7},
    "Mejor lluvia única de la grilla": None,
    "Temporal de 75 mm repartido en 3 días (1–3/3)": {"2026-03-01": 25, "2026-03-02": 25, "2026-03-03": 25},
}
best = grid.loc[grid["pred_clima"].idxmax()]
escenarios["Mejor lluvia única de la grilla"] = {best["fecha"]: best["mm"]}

# respuesta máxima posible: suelo vacío y lluvia que lo llena (ΔW = W)
doy_best = best["fecha"].dayofyear
A_max = ap.response_A(np.array([doy_best]), np.array([0.0]), params)[0]
H_max = A_max * np.log1p(params.W)

summary = []
for name, extra in escenarios.items():
    res, pc, pk = run(extra, name)
    summary.append({"escenario": name, "pred_9_16_mar_%": 100 * pc, "pred_conteos_%": 100 * pk,
                    "D50_err_d": res["D50_err_modelo_d"], "RMSE_pp": res["curva_RMSE_modelo_pp"],
                    "skill_clima": res["skill_clima"], "AUC_clima": res["AUC_clima"],
                    "skill_conteos": res["skill_conteos"], "AUC_conteos": res["AUC_conteos"],
                    "regla_3pct": res["regla_3pct_flushes"]})
S = pd.DataFrame(summary)

# cuánto tendría que aumentar la respuesta de marzo para reproducir el flush
F0 = float(counts.loc[counts["Fecha"] == FLUSH_INI, "F_obs"].iloc[0])
H_needed = -np.log(1 - obs / (1 - F0))

pd.set_option("display.width", 220)
print(f"Observado 9–16 mar: {100*obs:.1f} % de la campaña (F0 = {100*F0:.1f} %)")
print("\nPronóstico solo con clima para 9–16 mar (% de la campaña), por fecha y monto de la lluvia agregada:")
print((100 * grid.pivot(index="mm", columns="fecha", values="pred_clima")).round(1)
      .rename(columns=lambda c: c.strftime("%d/%m")).to_string())
print(f"\nTope del modelo: una lluvia que llena el suelo seco el {best['fecha']:%d/%m} da hazard {H_max:.3f}; "
      f"para {100*obs:.1f} % hace falta {H_needed:.3f} → la respuesta de marzo tendría que ser "
      f"{H_needed / H_max:.1f} veces mayor.")
print("\nEscenarios:")
print(S.round(3).to_string(index=False))

if len(sys.argv) > 1:
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    ramp = ["#a8cdf3", "#78afec", "#4f93e2", "#2a78d6", "#1d5fb3", "#14468a", "#0d2f5f"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for i, f in enumerate(FECHAS):
        g = grid[grid["fecha"] == f]
        ax.plot(g["mm"], 100 * g["pred_clima"], "-o", color=ramp[i], lw=2, ms=6, mec=SURF, mew=1.2)
        ax.text(g["mm"].iloc[-1] + 1.5, 100 * g["pred_clima"].iloc[-1], f"{f:%d/%m}", va="center", fontsize=8.5, color=INK2)
    ax.axhline(100 * obs, color="#eb6834", lw=2, ls="--")
    ax.text(5, 100 * obs - 0.6, f"observado 9–16 mar: {100*obs:.1f} %", color=INK, fontsize=9, va="top")
    ax.set_xlabel("Lluvia agregada en un día (mm), sobre la serie de la estación")
    ax.set_ylabel("Emergencia pronosticada 9–16 mar\n(% de la campaña)")
    ax.set_ylim(0, 23); ax.set_xlim(0, 86)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Cada línea es la fecha de la lluvia no medida (24/2 a 7/3)", loc="left", fontsize=10, color=INK)
    fig.suptitle("Ninguna lluvia única alcanza el flush de marzo en el modelo", x=0.02, ha="left", fontsize=12, color=INK)
    fig.savefig(sys.argv[1], dpi=160, bbox_inches="tight", facecolor=SURF)
    print("figura:", sys.argv[1])
