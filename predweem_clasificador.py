# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Clasificador oficial de patrones (1977–2015)
# ===============================================================
# Usa modelo .pkl con centroides Early–Intermediate–Late–Extended
# Calcula EMERREL si hay EMEAC
# Clasifica automáticamente un año nuevo
# Exporta distancias, percentiles y patrón final
# Genera gráfico comparativo opcional
# ===============================================================

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# A) Utilidades internas
# ---------------------------------------------------------------

def get_percentile_day(x, y, p):
    """
    Retorna el día juliano donde la curva acumulada alcanza un percentil p.
    """
    y = np.clip(y, 0, 1)
    if all(y < p):
        return np.nan
    return np.interp(p, y, x)


# ---------------------------------------------------------------
# B) Cargar modelo .pkl
# ---------------------------------------------------------------

def cargar_modelo_pkl(path_pkl):
    """
    Carga centroides (Early–Intermediate–Late–Extended)
    guardados en predweem_model_centroides.pkl
    """
    with open(path_pkl, "rb") as f:
        model = pickle.load(f)
    return model["centroides"]


# ---------------------------------------------------------------
# C) Procesar archivo de entrada (EMERREL o EMEAC)
# ---------------------------------------------------------------

def preparar_curva(df):
    """
    Acepta dataframe con:
    - Columna 0: JD (día juliano)
    - Columna 1: EMERREL *o* EMEAC (%)

    Devuelve:
    - jd (vector día juliano)
    - emer (curva acumulada normalizada 0–1)
    - rr   (emergencia relativa diaria)
    """

    jd = df.iloc[:,0].astype(float).values
    y  = df.iloc[:,1].astype(float).values

    # Detectar si es ya EMERREL
    if np.max(y) <= 1:
        # Podría ser EMERREL
        rr = y
        emer = np.cumsum(rr)
        emer = emer / emer.max()
    else:
        # Es EMEAC (acumulado %). Convertir:
        emer = y / np.max(y)
        rr = np.diff(emer, prepend=0)     # EMERREL diaria
        rr = np.clip(rr, 0, None)

    return jd, emer, rr


# ---------------------------------------------------------------
# D) Clasificación usando centroides
# ---------------------------------------------------------------

def clasificar_con_modelo(df_new, centroides):
    """
    Clasifica un año a partir del dataframe df_new con 2 columnas:
    [JD, EMERREL] o [JD, EMEAC]

    Devuelve:
      - patrón final
      - vector con (JD25, JD50, JD75, JD95)
      - diccionario de distancias
    """

    jd, emer, rr = preparar_curva(df_new)

    # Percentiles
    JD25 = get_percentile_day(jd, emer, 0.25)
    JD50 = get_percentile_day(jd, emer, 0.50)
    JD75 = get_percentile_day(jd, emer, 0.75)
    JD95 = get_percentile_day(jd, emer, 0.95)

    vector = np.array([JD25, JD50, JD75, JD95])

    # Distancias a centroides
    dist = {
        pat: np.linalg.norm(vector - centroides.loc[pat].values)
        for pat in centroides.index
    }

    mejor_patron = min(dist, key=dist.get)

    return mejor_patron, vector, dist


# ---------------------------------------------------------------
# E) Gráfico comparativo (opcional)
# ---------------------------------------------------------------

def graficar_comparacion(jd, emer, centroides, out_path=None):
    """
    Grafica la curva EMEAC del año y los centroides (JD25–95).
    """
    plt.figure(figsize=(10,6))
    plt.plot(jd, emer, color="orange", label="Año evaluado", linewidth=2)

    for pat in centroides.index:
        JD25, JD50, JD75, JD95 = centroides.loc[pat].values
        plt.scatter([JD25,JD50,JD75,JD95], [0.25,0.50,0.75,0.95], label=pat)

    plt.xlabel("Día juliano")
    plt.ylabel("Emergencia acumulada (0–1)")
    plt.title("Comparación de curva del año vs centroides")
    plt.legend()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------
# F) Uso directo desde terminal (opcional)
# ---------------------------------------------------------------

if __name__ == "__main__":

    print("\n=== 🌾 PREDWEEM Clasificador ===\n")

    # Rutas de ejemplo (ajustar al usar)
    modelo = "predweem_model_centroides.pkl"
    archivo_ano = "AVEFA_resultados_rango_EMERREL.csv"

    print("Cargando modelo...")
    centroides = cargar_modelo_pkl(modelo)

    print("Cargando año a clasificar...")
    df_ano = pd.read_csv(archivo_ano)

    print("Clasificando...")
    patron, vector, dist = clasificar_con_modelo(df_ano, centroides)

    print("\n🔍 RESULTADOS")
    print("----------------------------------")
    print("Patrón asignado:", patron)
    print("Vector (JD25, JD50, JD75, JD95):")
    print(vector)
    print("\nDistancias:")
    print(dist)

    # Gráfico
    print("\nGenerando gráfico comparativo...")
    jd, emer, rr = preparar_curva(df_ano)
    graficar_comparacion(jd, emer, centroides,
                         out_path="grafico_comparativo.png")

    print("\n📄 Guardado: grafico_comparativo.png")
    print("\n=== Listo ===")
