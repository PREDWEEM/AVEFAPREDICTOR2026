# predweem_ann_loader.py
import numpy as np
import pandas as pd

class ANN:
    """
    Carga la red neuronal entrenada (IW, LW, bias) desde archivos locales .npy
    y permite predecir EMERREL y EMEAC a partir de JD, TMIN, TMAX, Prec.
    """

    def __init__(self):
        # Pesos y biases guardados en el repo
        self.IW  = np.load("IW.npy")
        self.bIW = np.load("bias_IW.npy")
        self.LW  = np.load("LW.npy")
        self.bO  = np.load("bias_out.npy")

        if self.LW.ndim == 1:
            self.LW = self.LW.reshape(1, -1)

        # Aseguramos escalar a float
        self.bO = float(self.bO if np.ndim(self.bO) == 0 else np.ravel(self.bO)[0])

        # Mismos rangos de entrada que en tu app AVEFA
        # Orden: [JD, TMIN, TMAX, Prec]
        self.input_min = np.array([1, -7, 0, 0], dtype=float)
        self.input_max = np.array([300, 25.5, 41, 84], dtype=float)
        self._den = np.maximum(self.input_max - self.input_min, 1e-9)

    def _tansig(self, x):
        return np.tanh(x)

    def _normalize(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / self._den - 1

    def predict_emerrel(self, df_meteo: pd.DataFrame):
        """
        df_meteo debe tener columnas: JD, TMIN, TMAX, Prec.
        Devuelve:
        - emerrel: emergencia relativa diaria (0–1)
        - emeac: emergencia acumulada normalizada (0–1)
        """
        X = df_meteo[["JD", "TMIN", "TMAX", "Prec"]].to_numpy(float)
        Xn = self._normalize(X)

        z1 = Xn @ self.IW + self.bIW
        a1 = self._tansig(z1)

        z2 = (a1 @ self.LW.T).ravel() + self.bO
        y  = self._tansig(z2)          # [-1..1]
        emerrel = (y + 1) / 2          # → [0..1]
        emerrel = np.clip(emerrel, 0, 1)

        emeac = emerrel.cumsum()
        if emeac.max() > 0:
            emeac = emeac / emeac.max()

        return emerrel, emeac
