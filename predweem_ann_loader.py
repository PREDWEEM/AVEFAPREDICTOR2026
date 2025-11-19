# predweem_ann_loader.py
# placeholder ANN loader
import numpy as np
class ANN:
    def __init__(self):
        self.IW = np.load("IW.npy")
        self.bIW = np.load("bias_IW.npy")
        self.LW = np.load("LW.npy")
        self.bO = np.load("bias_out.npy")
    def predict_emerrel(self, df):
        return None, None
