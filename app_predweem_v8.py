# ===============================================================
# 🌾 PREDWEEM v8.5 — AVEFA Predictor 2026 
# ANN + Centroides + Reglas agronómicas avanzadas (Early/Interm/Ext/Late)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ---------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM v8.5 — AVEFA 2026", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 PREDWEEM v8.5 — AVEFA Predictor 2026")
st.subheader("ANN + Centroides + Reglas fisiológicas avanzadas")

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()


# ===============================================================
# FUNCIONES AUXILIARES
# ===============================================================

def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

def rmse_curvas(y_pred, y_obs):
    y_pred = np.asarray(y_pred, float)
    y_obs  = np.asarray(y_obs, float)
    return float(np.sqrt(np.mean((y_pred - y_obs) ** 2)))


# ===============================================================
# COMPUTO DE PERCENTILES
# ===============================================================
def _compute_jd_percentiles(jd, emerac, qs=(0.25, 0.5, 0.75, 0.95)):
    jd = np.asarray(jd, float)
    emer = np.asarray(emerac, float)

    order = np.argsort(jd)
    jd = jd[order]
    emer = emer[order]

    if emer.max() <= 0:
        return None

    y = emer / emer.max()
    return np.array([np.interp(q, y, jd) for q in qs], float)


# ===============================================================
# CARGA HISTÓRICA + CENTROIDES
# ===============================================================
def _load_curves_emereac():
    curvas = {}
    try:
        xls1 = pd.ExcelFile(BASE / "emergencia_acumulada_interpolada 1977-1998.xlsx")
        for sh in xls1.sheet_names:
            df = pd.read_excel(xls1, sh)
            year = int(sh.split("_")[-1])
            curvas[year] = df[["JD", "EMERAC"]]
    except: pass

    try:
        xls2 = pd.ExcelFile(BASE / "emergencia_2000_2015_interpolada.xlsx")
        for sh in xls2.sheet_names:
            df = pd.read_excel(xls2, sh)
            year = int(sh.split("_")[-1])
            curvas[year] = df[["JD", "EMERAC"]]
    except: pass

    return curvas

def _assign_labels_from_centroids(curvas, C):
    regs=[]
    for year, df in curvas.items():
        vals=_compute_jd_percentiles(df["JD"], df["EMERAC"])
        if vals is None: continue
        d25,d50,d75,d95=vals
        v=np.array(vals)
        d=np.linalg.norm(C.values - v,axis=1)
        patron=C.index[np.argmin(d)]
        regs.append({"anio":year,"patron":patron,
                     "JD25":d25,"JD50":d50,"JD75":d75,"JD95":d95})
    return pd.DataFrame(regs)

@st.cache_resource
def load_centroides_y_historia():
    cent = joblib.load(BASE/"predweem_model_centroides.pkl")
    C = cent["centroides"]
    curvas = _load_curves_emereac()
    labels = _assign_labels_from_centroids(curvas, C)

    rep={}
    for pat in C.index:
        sub=labels[labels["patron"]==pat]
        if len(sub)==0: continue
        vc=C.loc[pat][["JD25","JD50","JD75","JD95"]].values.astype(float)
        M=sub[["JD25","JD50","JD75","JD95"]].values.astype(float)
        best=np.argmin(np.linalg.norm(M-vc,axis=1))
        rep[pat]=int(sub.iloc[best]["anio"])
    return C, labels, rep, curvas


# ===============================================================
# ANN
# ===============================================================

class PracticalANNModel:
    def __init__(self,IW,bIW,LW,bLW):
        self.IW=IW; self.bIW=bIW; self.LW=LW; self.bLW=bLW
        self.input_min=np.array([1,0,-7,0])
        self.input_max=np.array([300,41,25.5,84])
    def normalize(self,X):
        return 2*(X-self.input_min)/(self.input_max-self.input_min)-1
    def predict(self,X):
        Xn=self.normalize(X)
        emer=[]
        for x in Xn:
            z1=self.IW.T@x + self.bIW
            a1=np.tanh(z1)
            z2=self.LW@a1 + self.bLW
            emer.append(np.tanh(z2))
        emer=(np.array(emer)+1)/2
        emerac=np.cumsum(emer)
        emerrel=np.diff(emerac,prepend=0)
        return emerrel, emerac

@st.cache_resource
def load_ann():
    IW=np.load(BASE/"IW.npy")
    bIW=np.load(BASE/"bias_IW.npy")
    LW=np.load(BASE/"LW.npy")
    bLW=np.load(BASE/"bias_out.npy")
    return PracticalANNModel(IW,bIW,LW,bLW)

def postprocess_emergence(raw,smooth=True,window=3,clip_zero=True):
    e=np.maximum(raw,0) if clip_zero else raw
    if smooth and window>1:
        k=np.ones(window)/window
        e=np.convolve(e,k,mode="same")
    return e, np.cumsum(e)


# ===============================================================
# NUEVO: REGLAS AGRONÓMICAS COMPLETAS
# ===============================================================
def aplicar_reglas_agronomicas(JD_ini, JD25, JD50, JD75, JD95, patron_inicial):
    """
    Devuelve 'override' si alguna regla define el patrón.
    Si no hay override, devuelve None.
    """

    banda = JD75 - JD25

    # =======================
    # 🔵 EXTENDED (PRIORIDAD)
    # =======================
    if (50 <= JD_ini <= 80) and (JD50 > 150) and (banda > 120):
        return "Extended"

    # =======================
    # 🟢 EARLY
    # =======================
    if (JD_ini < 70) and (JD50 < 140) and (banda < 60):
        return "Early"

    # =======================
    # 🟡 INTERMEDIATE
    # =======================
    if (70 <= JD_ini <= 110) and (130 <= JD50 <= 160) and (70 <= banda <= 90):
        return "Intermediate"

    # =======================
    # 🔴 LATE
    # =======================
    if (JD_ini > 110) and (JD50 > 160):
        return "Late"

    return None


# ===============================================================
# CLASIFICACIÓN ANN + CENTROIDES + REGLAS
# ===============================================================
def clasificar_patron_desde_ann(dias, emerac, C):

    vals=_compute_jd_percentiles(dias, emerac)
    if vals is None: return None,None,None,None

    d25,d50,d75,d95=vals
    v=np.array(vals)

    # Distancias a centroides
    dists=np.linalg.norm(C.values - v, axis=1)
    w=1.0/(dists+1e-6)
    p=w/w.sum()
    prob_base={str(C.index[i]):float(p[i]) for i in range(len(C.index))}

    emerac=np.asarray(emerac)
    dias=np.asarray(dias)

    # JD_ini
    idx=np.where(emerac>0.01)[0]
    JD_ini=dias[idx[0]] if len(idx)>0 else np.inf

    # ============================================================
    # APLICAR REGLAS AGRONÓMICAS
    # ============================================================
    override = aplicar_reglas_agronomicas(JD_ini, d25, d50, d75, d95, None)

    if override in C.index:
        # Sobreescribir patrón
        patron_final = override

        # Modificar ranking para darle prob ~1
        d_mod = dists.copy()
        idxp=list(C.index).index(patron_final)
        d_mod[idxp] = -1.0

        w=1/(d_mod - d_mod.min() + 1e-6)
        p=w/w.sum()
        prob_mod={str(C.index[i]):float(p[i]) for i in range(len(C.index))}
        return patron_final, vals, d_mod, prob_mod

    # ============================================================
    # SI NO HAY REGLAS — usar centroide más cercano
    # ============================================================
    idx_min=int(np.argmin(dists))
    patron=str(C.index[idx_min])

    return patron, vals, dists, prob_base



# ===============================================================
# NORMALIZAR ARCHIVO METEO
# ===============================================================
def normalizar_meteo(df):
    df=df.copy()
    cols={c.lower():c for c in df.columns}

    def pick(*names):
        for n in names:
            for k,v in cols.items():
                if n in k: return v
        return None

    c_jd=pick("jd","julian","dia")
    c_tmax=pick("tmax")
    c_tmin=pick("tmin")
    c_prec=pick("prec","lluv")
    c_fecha=pick("fecha")

    if None in (c_jd,c_tmax,c_tmin,c_prec):
        raise ValueError("No se identifican JD/TMAX/TMIN/Prec")

    df["JD"]=pd.to_numeric(df[c_jd],errors="coerce")
    df["TMAX"]=pd.to_numeric(df[c_tmax],errors="coerce")
    df["TMIN"]=pd.to_numeric(df[c_tmin],errors="coerce")
    df["Prec"]=pd.to_numeric(df[c_prec],errors="coerce")
    df["Fecha"]=pd.to_datetime(df[c_fecha],errors="coerce") if c_fecha else None

    df=df.dropna(subset=["JD"])
    return df.sort_values("JD")


# ===============================================================
# SIDEBAR
# ===============================================================
with st.sidebar:
    st.header("Ajustes ANN")
    smooth=st.checkbox("Suavizar EMERREL",True)
    window=st.slider("Ventana",1,9,3)
    clip=st.checkbox("Cortar negativos",True)

# ===============================================================
# CARGA METEO
# ===============================================================
st.subheader("📤 Cargar archivo meteorológico")
up=st.file_uploader("CSV/XLSX",type=["csv","xlsx"])

modelo_ann=safe(load_ann,"Error cargando ANN")
if up is None: st.stop()

df_raw=pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
st.dataframe(df_raw,use_container_width=True)

try:
    df_meteo=normalizar_meteo(df_raw)
except Exception as e:
    st.error(e); st.stop()

if modelo_ann is None: st.stop()


# ===============================================================
# ANN → EMERREL / EMERAC
# ===============================================================
st.subheader("🔍 ANN → EMERREL / EMERAC")

df_ann=df_meteo.copy()
dias=df_ann["JD"].to_numpy()
X=df_ann[["JD","TMAX","TMIN","Prec"]].to_numpy()

raw_rel,raw_ac=modelo_ann.predict(X)
emerrel,emerac=postprocess_emergence(raw_rel,smooth,window,clip)

df_ann["EMERREL"]=emerrel
df_ann["EMERAC"]=emerac


# ===============================================================
# CLASIFICACIÓN COMPLETA
# ===============================================================
st.subheader("🌱 Clasificación del patrón")

C,labels,rep_year,curvas_hist=safe(lambda:load_centroides_y_historia(),
                                   "Error cargando centroides/histórico")

if C is None: st.stop()

patron,vals,dists,probs=clasificar_patron_desde_ann(dias,emerac,C)

if patron is None:
    st.error("No se pudo clasificar"); st.stop()

st.success(f"Patrón resultante: **{patron}**")
st.write("Probabilidades:",probs)

dist_table=pd.DataFrame({
    "Patrón":list(C.index),
    "Distancia":dists,
    "Prob": [probs[str(p)] for p in C.index]
}).sort_values("Distancia")

st.dataframe(dist_table,use_container_width=True)

if patron in rep_year:
    st.info(f"Año representativo: **{rep_year[patron]}**")


# ===============================================================
# DESCARGA
# ===============================================================
st.download_button("📥 Descargar EMERREL/EMERAC",
                   df_ann.to_csv(index=False).encode("utf-8"),
                   "emergencia_ann_v85.csv")




