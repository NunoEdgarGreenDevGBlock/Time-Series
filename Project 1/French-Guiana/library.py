import statsmodels.tsa.seasonal as stats
import numpy as np
import scipy as sp


def get_trend(series, period=365):
    trend = series.rolling(period).mean()

    return trend


def remove_trend(series, period=365):
    trend = get_trend(series, period=period)
    stationary_series = series - trend

    return stationary_series


def noise(series, period=365, cut_off=2.1):
    fft = sp.fftpack.fft(series.values)
    fft_freq = sp.fftpack.fftfreq(len(series), 1 / period)

    cleaned_fft = fft.copy()
    cleaned_fft[np.abs(fft_freq) < cut_off] = 0

    noise = np.real(sp.fftpack.ifft(cleaned_fft))

    return noise


def no_noise(series, period=365):
    no_noise = series - noise(series, period=period)

    return no_noise


def autocorrelation_pd(series):
    rho = []
    for k in range(len(series)):
        rho.append(series.autocorr(k))

    return np.array(rho)


def autocorrelation(series):
    # Source: https://iopscience.iop.org/article/10.1088/0026-1394/47/5/012/pdf
    mu = np.nanmean(series)
    norm = np.var(series) * len(series)
    aux = np.array(series.values - mu)
    rho = [aux.dot(aux)]
    for k in range(1, len(series)):
        conv = np.array(aux[k:].dot(aux[:-k]))
        rho.append(conv)

    return np.array(rho / norm)


def corrected_mean(series, rho):
    N = len(series)
    var = np.var(series)
    aux = np.array(list(range(1, N)))
    var_corrected = var / N * (1 + 2 * np.sum((N - aux) * rho[aux]) / N)
    var_naive = var / N

    return np.mean(series), np.sqrt(var_corrected), np.sqrt(var_naive)
