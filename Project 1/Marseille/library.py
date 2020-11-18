import numpy as np


def empricial_auto_correlation(series, t):
    # https://itp.tugraz.at/~evertz/Computersimulations/cs2020.pdf Page 22

    N_t = 1 / (len(series) - t)

    x_bar = np.sum(series) * N_t if t == 0 else np.sum(series[:-t]) * N_t
    y_bar = np.sum(series[t:]) * N_t

    x_term = series - x_bar if t == 0 else series[:-t] - x_bar
    y_term = series[t:] - y_bar

    rho = np.sum(x_term * y_term) / \
        np.sqrt(np.nansum(x_term**2) * np.nansum(y_term**2))

    return rho
