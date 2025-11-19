# predweem_predictor.py
import pandas as pd
import joblib

MODEL_PATH = "predweem_meteo2patron.pkl"

def extraer_features(df):
    df2 = df.copy()
    df2["Tmed"] = (df2["TMIN"] + df2["TMAX"]) / 2
    feats = {
        "Tmin_mean": df2["TMIN"].mean(),
        "Tmax_mean": df2["TMAX"].mean(),
        "Tmed_mean": df2["Tmed"].mean(),
        "Prec_total": df2["Prec"].sum(),
        "Prec_days_10mm": (df2["Prec"] >= 10).sum(),
        "Tmed_FM": df2[df2["JD"] <= 121]["Tmed"].mean(),
        "Prec_FM": df2[df2["JD"] <= 121]["Prec"].sum()
    }
    return feats

def predecir_patron(df):
    df = df.rename(columns={
        "jd": "JD", "julian_days": "JD",
        "tmin": "TMIN", "tmax": "TMAX",
        "prec": "Prec"
    })

    df = df[["JD","TMIN","TMAX","Prec"]].dropna()

    X = pd.DataFrame([extraer_features(df)])

    model = joblib.load(MODEL_PATH)

    probas = model.predict_proba(X)[0]
    clases = model.classes_
    patron = clases[probas.argmax()]

    return {
        "clasificacion": patron,
        "probabilidades": dict(zip(clases, map(float, probas)))
    }
r
