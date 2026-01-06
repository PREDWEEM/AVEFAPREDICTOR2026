import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
OUT = Path("meteo_daily.csv")

# Definimos el inicio de la cuenta (Día 1)
START = datetime(2026, 1, 1)

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

        fecha_dt = pd.to_datetime(fecha_str)
        
        # --- CÁLCULO DEL DÍA JULIANO (Base 01/01/2026 = 1) ---
        # Calculamos la diferencia de días y sumamos 1
        dia_juliano = (fecha_dt - START).days + 1
        
        rows.append({
            "Fecha": fecha_dt,
            "Juliano": dia_juliano,
            "TMAX": to_float(tmax),
            "TMIN": to_float(tmin),
            "Prec": to_float(prec),
        })

    df = pd.DataFrame(rows).sort_values("Fecha")
    return df

def update_file():
    today = datetime.utcnow().date()

    if today < START.date():
        print(f"⏳ Antes del {START.date()} → no se actualiza.")
        return

    if today == START.date():
        if OUT.exists():
            OUT.unlink()
            print(f"🆕 Archivo reiniciado para el ciclo 2026.")

    df_new = fetch_meteobahia()

    if OUT.exists():
        df_old = pd.read_csv(OUT, parse_dates=["Fecha"])
        df_all = pd.concat([df_old, df_new]).drop_duplicates("Fecha").sort_values("Fecha")
    else:
        df_all = df_new

    # Asegurar orden de columnas
    cols = ["Fecha", "Juliano", "TMAX", "TMIN", "Prec"]
    df_all = df_all[cols]

    df_all.to_csv(OUT, index=False)
    print(f"[OK] {len(df_all)} registros. Hoy es Día Juliano: {(pd.to_datetime(today) - START).days + 1}")

if __name__ == "__main__":
    update_file()
