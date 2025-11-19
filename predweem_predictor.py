# predweem_predictor.py
import pandas as pd
import joblib

MODEL = "/mnt/data/predweem_meteo2patron.pkl"

def extraer_features(df):
    df["Tmed"] = (df["TMIN"] + df["TMAX"]) / 2
    feats = {}
    feats["Tmin_mean"] = df["TMIN"].mean()
    feats["Tmax_mean"] = df["TMAX"].mean()
    feats["Tmed_mean"] = df["Tmed"].mean()
    feats["Prec_total"] = df["Prec"].sum()
    feats["Prec_days_10mm"] = (df["Prec"]>=10).sum()
    feats["Tmed_FM"] = df[df["JD"]<=121]["Tmed"].mean()
    feats["Prec_FM"] = df[df["JD"]<=121]["Prec"].sum()
    return feats

def predecir_patron(path_csv):
    df = pd.read_csv(path_csv)
    df = df.rename(columns={"Julian_days":"JD","tmin":"TMIN","tmax":"TMAX","prec":"Prec"})
    df = df[["JD","TMIN","TMAX","Prec"]].dropna()

    X = pd.DataFrame([extraer_features(df)])

    model = joblib.load(MODEL)
    probas = model.predict_proba(X)[0]
    clases = model.classes_
    pred = clases[probas.argmax()]

    return {
        "clasificacion": pred,
        "probabilidades": dict(zip(clases, map(float, probas)))
    }

if __name__=="__main__":
    print(predecir_patron("/mnt/data/meteo_history.csv"))

