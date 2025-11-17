# PREDWEEM – Clasificador de patrones de emergencia (Early–Intermediate–Late–Extended)
# Basado en percentiles JD25–JD50–JD75–JD95
# Listo para subir a GitHub

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

def get_percentile_day(x, y, p):
    y = np.clip(y,0,1)
    if all(y < p):
        return np.nan
    return np.interp(p, y, x)

def construir_modelo(file1, file2):
    data1 = pd.read_excel(file1, sheet_name=None)
    data2 = pd.read_excel(file2, sheet_name=None)

    registros = []
    for dataset in [data1, data2]:
        for sheet, df in dataset.items():
            x = df.iloc[:,0].values.astype(float)
            for col in df.columns[1:]:
                y = df[col].astype(float).values
                registros.append({
                    "ID": f"{sheet}_{col}",
                    "JD25": get_percentile_day(x,y,0.25),
                    "JD50": get_percentile_day(x,y,0.50),
                    "JD75": get_percentile_day(x,y,0.75),
                    "JD95": get_percentile_day(x,y,0.95)
                })

    df_hist = pd.DataFrame(registros)
    X = df_hist[["JD25","JD50","JD75","JD95"]].fillna(df_hist.mean())
    Z = linkage(X, method="ward")
    df_hist["Cluster4"] = fcluster(Z, 4, criterion="maxclust")

    cluster_centers = df_hist.groupby("Cluster4")[["JD25","JD50","JD75","JD95"]].mean()
    orden = cluster_centers.sort_values("JD50").index.tolist()

    name_map = {
        orden[0]: "Early",
        orden[1]: "Intermediate",
        orden[2]: "Late",
        orden[3]: "Extended"
    }

    df_hist["Pattern"] = df_hist["Cluster4"].map(name_map)
    centroides = df_hist.groupby("Pattern")[["JD25","JD50","JD75","JD95"]].mean()

    return centroides

def clasificar_nuevo_anio(df_new, centroides):
    jd = df_new.iloc[:,0].values.astype(float)
    rr = df_new.iloc[:,1].values.astype(float)

    emer = np.cumsum(rr)
    emer = emer / emer.max()

    JD25 = get_percentile_day(jd, emer, 0.25)
    JD50 = get_percentile_day(jd, emer, 0.50)
    JD75 = get_percentile_day(jd, emer, 0.75)
    JD95 = get_percentile_day(jd, emer, 0.95)

    vector = np.array([JD25, JD50, JD75, JD95])

    dist = {pat: np.linalg.norm(vector - centroides.loc[pat].values)
            for pat in centroides.index}

    mejor = min(dist, key=dist.get)

    return mejor, vector, dist

if __name__ == "__main__":
    print("=== Clasificador de patrones PREDWEEM ===")
    # El usuario debe ajustar las rutas de los archivos históricos aquí:
    # centroides = construir_modelo("CURVAS AVEFA 1977-1998.xlsx", "avefa 2000-2015.xlsx")
    # df_nuevo = pd.read_excel("2014.xlsx", header=None)
    # print(clasificar_nuevo_anio(df_nuevo, centroides))
