import numpy as np

#X.shape = (sample_size, feature_size)
#t is a int and t >= 0
def autocorrelation(X, t):
    x_mean = np.mean(X, axis=0)
    x_shift = X - x_mean
    x_var = np.mean(np.square(x_shift), axis=0)

    sample_size = X.shape[0]
    

    x1 = x_shift[0:sample_size-t]
    x2 = x_shift[t:sample_size]

    y = x1*x2

    result = np.mean(y, 0)/x_var

    return result

if __name__ == '__main__':
    import os
    import sys
    import matplotlib.pyplot as plt
    from matplotlib import colors
    sys.path.append('D:/shan/repo')
    from MTS_performance.datasets.multivariate_time_series_data.get import get_electricity, get_exchange_rate, get_solar_energy, get_traffic
    electricity = get_solar_energy()
    #print(type(electricity))
    #exit()
    print(electricity.shape)
    ar_list = []
    for t in range(0, 400):
        ar = autocorrelation(electricity, t)
        ar_list.append(ar)
    x = list(range(0, 400))
    y = np.array(ar_list)

    y = np.mean(y, axis=1)
    plt.plot(x, y)
    plt.show()
    exit()
    for i in range(2, y.shape[0]-2):
        if y[i-2]< y[i-1]and y[i-1] < y[i] and y[i] > y[i+1] and y[i+1]>y[i+2]:
            print(i)



    #exit()
    print(y.shape)
    cs = ['black', 'maroon', 'saddlebrown', 'darkorange', 'limegreen', 'dodgerblue', 'cyan', 'olive']
    for i in range(8):
        y_t = y[:, i]
        plt.plot(x, y_t, ls='-', lw=1, c=cs[i], label=str(i))
    plt.legend(loc='upper right')
    plt.show()
    

