import numpy as np

def MSE(Y1, Y2):
    t = Y1 - Y2
    return np.mean(np.square(t))
