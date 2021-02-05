import numpy as np

def MAE(Y1, Y2):
    t = Y1 - Y2
    return np.mean(np.abs(t))

