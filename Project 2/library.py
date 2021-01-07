import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal.windows import boxcar
from scipy.signal import convolve, ellip, filtfilt, freqz


def load_data(path):
    df = pd.read_csv(path, header=12)
    df.insert(0, 'Date', pd.to_datetime(
        {'year': df['YEAR'], 'month': df['MO'], 'day': df['DY']}
    ))
    df = df.drop(columns=['LAT', 'LON', 'YEAR', 'MO', 'DY'])
    df.rename(
        columns={'PS': 'Pressure', 'QV2M': 'Humidity', 'T2M': 'Temperature'},
        inplace=True)

    return df


def apply_boxcar(series, T):
    b = boxcar(T) / T
    return convolve(series, b, mode='same', method='direct')


def apply_filter(series, b, a):
    return filtfilt(b, a, series)


def get_elliptic_filter(N, rp, rs, Wn):
    Wn /= 365  # 1/year
    b, a = ellip(N, rp, rs, Wn, 'low', analog=False, output='ba', fs=1)

    return b, a


def plot_series(df, series, labels=None):
    _, ax = plt.subplots(figsize=(9.0, 3))

    for serie, label in zip(series, labels):
        ax.plot(df['Date'], df[serie], label=label, linewidth=0.75)

    ax.set(xlabel=r'Date'.format(df['Date'][0].date()),
           ylabel=fr'Temperature [°C]')
    ax.grid()
    plt.tight_layout()

    return ax


def plot_filter(b, a, Wn=None):
    w, h = freqz(b, a, worN=2**10, fs=1)

    _, ax = plt.subplots()
    ax.plot(w, conv_to_db(abs(h)), label='Elliptic Filter')

    if Wn is not None:
        ax.axvline(Wn / 365, color='orange')

    ax.set_title('Elliptic Filter')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_xlabel('Frequency [1/day]')
    ax.grid()
    ax.legend()

    return ax


def plot_spectrum(df, series, labels):
    _, ax = plt.subplots()

    for serie, label in zip(series, labels):
        N = len(df[serie])
        psd = fft(df[serie].values)[:N // 2]**2
        freqs = fftfreq(N, 1)[:N // 2]
        ax.semilogx(freqs, conv_to_db(psd), label=label)

    ax.set_title('Frequency Domain')
    ax.set_ylabel('PSD [dB]')
    ax.set_xlabel('Frequency [1/day]')
    ax.grid()
    ax.legend()
    plt.tight_layout()

    return ax


def conv_to_db(x, x_0=1):
    return 20 * np.log10(x / x_0)
