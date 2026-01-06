import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
OUT = Path("meteo_daily.csv")

# Definimos el inicio de la cuenta (Día 1) como Timestamp de Pandas para evitar errores de tipo
START = pd.Timestamp("2026-01-01")

def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

def fetch_meteobahia():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    r = requests.get(URL, headers=headers, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    rows = []
    for d in root.findall(".//forecast/tabular/day"):
        fecha_str = d.find("fecha").get("value")
        tmax  = d.find("tmax").get("value")
        tmin  = d.find("tmin").get("value")
        prec  = d.find("precip").get("value")

        rows.append({
            "Fecha": pd.to_datetime(fecha_str),
            "TMAX": to_float(tmax),
            "TMIN": to_float(tmin),
            "Prec": to_float(prec),
        })

    df = pd.DataFrame(rows)
    return df

def update_file():
    # Usar Timestamp para consistencia
    today = pd.Timestamp(datetime.utcnow().date())

    if today < START:
        print(f"⏳ Antes del {START.date()} → no se actualiza.")
        return

    if today == START:
        if OUT.exists():
            OUT.unlink()
            print(f"🆕 Archivo reiniciado para el ciclo 2026.")

    # 1. Obtener datos nuevos
    df_new = fetch_meteobahia()

    # 2. Combinar con datos existentes si existen
    if OUT.exists():
        df_old = pd.read_csv(OUT, parse_dates=["Fecha"])
        df_all = pd.concat([df_old, df_new]).drop_duplicates("Fecha")
    else:
        df_all = df_new

    # 3. CÁLCULO VECTORIAL DEL DÍA JULIANO (Garantiza exactitud)
    # Convertimos a datetime por si acaso y restamos el inicio
    df_all["Fecha"] = pd.to_datetime(df_all["Fecha"])
    df_all["Juliano"] = (df_all["Fecha"] - START).dt.days + 1

    # 4. Ordenar y guardar
    cols = ["Fecha", "Juliano", "TMAX", "TMIN", "Prec"]
    df_all = df_all[cols].sort_values("Fecha")

    df_all.to_csv(OUT, index=False)
    
    juliano_hoy = (today - START).days + 1
    print(f"[OK] {len(df_all)} registros en {OUT.name}")
    print(f"📌 Info: Hoy es {today.date()} (Día Juliano: {juliano_hoy})")

if __name__ == "__main__":
    update_file()
