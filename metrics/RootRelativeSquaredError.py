import numpy as np

def RRSE(y, y_pred):
    #
    # The formula is:
    #           K.sqrt(K.sum(K.square(y_true - y_pred)))     
    #    RSE = -----------------------------------------------
    #           K.sqrt(K.sum(K.square(y_true_mean - y_true)))       
    #
    #           K.sqrt(K.sum(K.square(y_true - y_pred))/(N-1))
    #        = ----------------------------------------------------
    #           K.sqrt(K.sum(K.square(y_true_mean - y_true)/(N-1)))
    #
    #
    #           K.sqrt(K.mean(K.square(y_true - y_pred)))
    #        = ------------------------------------------
    #           K.std(y_true)
    #
    # return np.sqrt(
    #     np.sum(np.square(y - y_pred))
    #     /
    #     np.sum(np.square(np.mean(y) - y))
    # )
    num = np.sqrt(np.mean(np.square(y - y_pred)))
    den = np.std(y)
    return num/den


if __name__ == '__main__':
    y = np.random.randn(10000, 3)
    y_pred = np.random.randn(10000, 3)
    result = RRSE(y, y_pred)
    print(result)