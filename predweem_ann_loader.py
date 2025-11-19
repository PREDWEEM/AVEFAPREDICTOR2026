# predweem_ann_loader.py
import numpy as np
import pandas as pd
from io import BytesIO
from urllib.request import urlopen, Request

RAW = "https://raw.githubusercontent.com/PREDWEEM/AVEFA2/main"

def _fetch_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=20) as resp:
        return resp.read()

def load_npy(filename: str) -> np.ndarray:
    raw = _fetch_bytes(f"{RAW}/{filename}")
    return np.load(BytesIO(raw), allow_pickle=False)

class ANN:
    """Versión desacoplada del modelo PracticalANNModel del script AVEFA."""
    def __init__(self):
        IW  = load_npy("IW.npy")
        bIW = load_npy("bias_IW.npy")
        LW  = load_npy("LW.npy")
        bO  = load_npy("bias_out.npy")

        if LW.ndim == 1:
            LW = LW.reshape(1, -1)

        self.IW, self.bIW, self.LW = IW, bIW, LW
        self.bO = float(bO if np.ndim(bO)==0 else np.ravel(bO)[0])

        # Normalización original
        self.input_min = np.array([1, -7, 0, 0], dtype=float)
        self.input_max = np.array([300, 25.5, 41, 84], dtype=float)
        self._den = np.maximum(self.input_max - self.input_min, 1e-9)

    def _tansig(self, x):
        return np.tanh(x)

    def _normalize(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2*(Xc - self.input_min)/self._den - 1

    def predict_emerrel(self, df_meteo):
        """
        df_meteo con columnas: JD, TMIN, TMAX, Prec
        Devuelve EMERREL diaria (incremento) + EMEAC acumulada.
        """
        X_real = df_meteo[["JD","TMIN","TMAX","Prec"]].to_numpy(float)
        Xn = self._normalize(X_real)

        z1 = Xn @ self.IW + self.bIW
        a1 = self._tansig(z1)
        z2 = (a1 @ self.LW.T).ravel() + self.bO
        y  = self._tansig(z2)          # [-1..1]
        y  = (y - (-1)) / (1 - (-1))   # → [0..1]  (denorm_out)
        emerrel = np.clip(y, 0, 1)

        emeac = emerrel.cumsum()
        emeac = emeac / max(emeac.max(), 1e-9)

        return emerrel, emeac

