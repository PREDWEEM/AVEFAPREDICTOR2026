"""Figura de validación 2026 (estilo Fig. 5 del manuscrito). Uso: python tools/figura_validacion_2026.py salida.png"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import avefa_pulsos as ap  # noqa: E402
from validar_2026 import evaluate, load_counts  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OBS, BLUE, ORANGE = "#8d8c87", "#2a78d6", "#eb6834"

out = sys.argv[1] if len(sys.argv) > 1 else "validacion_2026.png"
counts, total = load_counts()
meteo = ap.clean_meteo(pd.read_csv(ROOT / "meteo_daily.csv"))
res, v, sim, F_clim, _ = evaluate(ap.Params.load(ROOT / "pulsos_params.json"), counts, meteo, "ajustado")
d = sim.days
x0, x1 = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-09-16")
m = (d["Fecha"] >= x0) & (d["Fecha"] <= x1)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 9.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 3, 1.3], "hspace": 0.12})
fig.patch.set_facecolor(SURF)
for a in (a1, a2, a3):
    a.set_facecolor(SURF)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.spines[["top", "right"]].set_visible(False)
    a.set_axisbelow(True)

# (a) curva acumulada
a1.plot(d.loc[m, "Fecha"], 100 * F_clim[m.to_numpy()], color=OBS, lw=2, ls="--", label="Curva media (solo línea de base)")
a1.plot(d.loc[m, "Fecha"], 100 * d.loc[m, "F"], color=BLUE, lw=2, label="Simulada solo con clima desde el 1 de febrero")
a1.scatter(v["fin"], 100 * v["F_obs"], s=36, color=INK, zorder=3, edgecolor=SURF, lw=1.2, label="Observada 2026 (sin remoción)")
a1.set_ylabel("Emergencia acumulada\n(% de la campaña)")
a1.set_ylim(0, 102)
a1.legend(loc="lower right", frameon=False, fontsize=9)
a1.set_title(f"(a) Curva de la campaña · RMSE {res['curva_RMSE_modelo_pp']:.1f} pp (curva media {res['curva_RMSE_clim_pp']:.1f}) · "
             f"50 %: observado {res['D50_obs']}, modelo {res['D50_modelo']}, curva media {res['D50_clim']}",
             loc="left", fontsize=10, color=INK)

# (b) emergencia por intervalo y pronósticos
for _, r in v.iterrows():
    a2.bar(r["inicio"] + pd.Timedelta(hours=10), 100 * r["dF_obs"], width=r["dias"] - 0.85, align="edge",
           color="#d3d2cd", edgecolor="none")
mid = v["inicio"] + (v["fin"] - v["inicio"]) / 2
a2.plot(mid, 100 * v["pred_clima"], "-o", color=BLUE, lw=2, ms=6, mec=SURF, mew=1.2, label="Pronóstico solo con clima")
a2.plot(mid, 100 * v["pred_conteos"], "-o", color=ORANGE, lw=2, ms=6, mec=SURF, mew=1.2, label="Pronóstico con conteos")
a2.axhline(3, color=INK2, lw=1, ls=":")
a2.text(pd.Timestamp("2026-02-03"), 3.25, "umbral 3 %", ha="left", va="bottom", fontsize=8.5, color=INK2)
from matplotlib.patches import Patch
h, l = a2.get_legend_handles_labels()
h.append(Patch(color="#d3d2cd")); l.append("Observada por intervalo")
big = v.loc[v["dF_obs"].idxmax()]
a2.annotate(f"{100*big['dF_obs']:.1f} % en una semana\n(pronóstico {100*big['pred_clima']:.1f} %)",
            xy=(big["inicio"] + pd.Timedelta(days=3.5), 100 * big["dF_obs"]), xytext=(pd.Timestamp("2026-04-02"), 17.5),
            fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a2.set_ylabel("Emergencia en el intervalo\n(% de la campaña)")
a2.set_ylim(0, 23)
a2.legend(h, l, loc="upper right", frameon=False, fontsize=9)
a2.set_title(f"(b) Intervalo siguiente a cada conteo, con la lluvia registrada hasta el conteo · "
             f"AUC flushes {res['AUC_ref_clima']:.2f}→{res['AUC_clima']:.2f} (clima), "
             f"{res['AUC_ref_conteos']:.2f}→{res['AUC_conteos']:.2f} (conteos)", loc="left", fontsize=10, color=INK)

# (c) lluvia
a3.bar(d.loc[m, "Fecha"], d.loc[m, "Prec"], width=0.8, color=BLUE, alpha=0.55)
a3.set_ylabel("Lluvia (mm)")
a3.set_title("(c) Lluvia diaria, estación SIGA–INTA Bordenave (4–10 jun: ECMWF provisional)", loc="left", fontsize=10, color=INK)
a3.xaxis.set_major_locator(mdates.MonthLocator())
MESES = ["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
a3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: MESES[mdates.num2date(x).month]))
a3.set_xlim(x0, x1)

fig.suptitle("Avena fatua, Bordenave 2026: modelo de pulsos (vector ajustado 1977–2015) frente a los conteos",
             x=0.07, ha="left", fontsize=12, color=INK, y=0.95)
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURF)
print("ok", out)
