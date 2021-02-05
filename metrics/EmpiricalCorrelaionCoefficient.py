import numpy as np

def CORR(y, y_pred):
    m = np.mean(y, 0)
    mp = np.mean(y_pred, 0)

    s0 = y - m
    s1 = y_pred - mp
    A = np.sum(s0*s1, 0)
    B = np.sum(np.square(s0), 0)
    C = np.sum(np.square(s1), 0)
    D = np.sqrt(B*C)
    t = D!=0
    E = A[t]/D[t]
    F = np.mean(E, 0)
    return F

if __name__ == '__main__':
    y = np.random.randn(10000, 3)
    y_pred = np.random.randn(10000, 3)
    result = CORR(y, y_pred)
    print(result)

