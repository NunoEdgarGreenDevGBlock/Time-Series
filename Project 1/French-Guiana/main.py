import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as tsa
import scipy.signal as signal
from fitting import polynomial_fit, get_polynomial_funcs, \
    eval_result
from library import remove_trend, autocorrelation, noise, corrected_mean, \
    no_noise

path = 'Temperature_dataset.csv'

# Read and tidy up dataset
df = pd.read_csv(path, header=12)
df.insert(0, 'Date', pd.to_datetime(
    {'year': df['YEAR'], 'month': df['MO'], 'day': df['DY']}
))
df = df.drop(columns=['LAT', 'LON', 'YEAR', 'MO', 'DY'])
df.rename(
    columns={'PS': 'Pressure', 'QV2M': 'Humidity', 'T2M': 'Temperature'},
    inplace=True)

# Plot Raw Data
_, ax = plt.subplots()
ax.plot(df.index, df['Temperature'], label='Raw Data')
ax.set(xlabel=r'Days since {} [days]'.format(df['Date'][0].date()),
       ylabel=fr'Temperature [°C]',
       title=fr'Raw Temperature Data in French-Guiana')

ax.grid()

# Remove the trend to obtain stationary signal
df['Stationary Temperature'] = remove_trend(df['Temperature'])
df['Residual Temperature'] = noise(df['Temperature'])
df['Noiseless Temperature'] = no_noise(df['Temperature'])
_, ax = plt.subplots()
ax.plot(df.index, df['Stationary Temperature'], label='Stationary Data')
ax.set(xlabel=r'Days since {} [days]'.format(df['Date'][0].date()),
       ylabel=fr'Temperature [°C]',
       title=fr'Trendless Temperature Data in French-Guiana')
ax.grid()

# Autocorrelation
df['Rho Own'] = autocorrelation(df['Temperature'])
df['Rho Stats'] = tsa.acf(df['Temperature'], nlags=len(df.index),
                          fft=False, missing='conservative')
df['Rho Stats Residual'] = tsa.acf(df['Residual Temperature'], nlags=len(
    df.index), fft=False, missing='conservative')
df['Rho Stats Stationary'] = tsa.acf(df['Stationary Temperature'], nlags=len(
    df.index), fft=False, missing='conservative')
alpha_005 = 1.96 / np.sqrt(len(df.index)) * np.ones(df.index.shape)
alpha_001 = 2.58 / np.sqrt(len(df.index)) * np.ones(df.index.shape)

_, ax = plt.subplots()
ax.plot(df.index, df['Rho Stats Stationary'],
        label='Autocorrelation Stationary')
ax.plot(df.index, df['Rho Stats Residual'], label='Autocorrelation Residual')
# ax.plot(df.index, df['Rho Stats'], label='Stats Autocorrelation',
#         marker='o', markersize=2)
# ax.plot(df.index, df['Rho Stats Stat'], label='Stats Autocorrelation Stat')
ax.plot(df.index, alpha_005, color='k', linewidth=0.5,
        label=fr'$95\%$ Confidence')
ax.plot(df.index, -alpha_005, color='k', linewidth=0.5)
ax.plot(df.index, alpha_001, color='k', linestyle='--', linewidth=0.5,
        label=fr'$99\%$ Confidence')
ax.plot(df.index, -alpha_001, color='k', linestyle='--', linewidth=0.5)
ax.set(xlabel=r'Lag $k$ [days]',
       ylabel=fr'Autocorrelation $\rho$ [1]',
       title=fr'Autocorrelation')
ax.grid()
ax.legend()

# Split Data into yearly batch sizes
df_yearly = pd.DataFrame(index=list(range(1, 366)))
num_years = int(len(df.index) / 365)

for i in range(num_years):
    col_name = 'Year {0:d}'.format(i + 1)
    df_yearly[col_name] = pd.Series(index=df_yearly.index,
                                    data=df['Temperature'].values[
                                         i * 365:    (i + 1) * 365])

# Calculate autocorrelation corrected mean for each "quasi stationary" year
means = []
for _, data in df_yearly.iteritems():
    rho = autocorrelation(data)
    means.append(corrected_mean(data, rho))
means = np.array(means).T

# Fit linear, quadratic, cubic and exponential
x = np.array(list(range(1, 21)))

_, ax = plt.subplots()
ax.errorbar(x, means[0], yerr=means[1], linestyle='none',
            capsize=5, marker='x', color='k', label='Data')

# Exponential fit converges to linear fit
labels = ['Linear Fit', 'Quadratic Fit', 'Cubic Fit']
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, label, color in zip(range(1, 4), labels, colors):
    p_opt, p, cov, _ = polynomial_fit(x, means[0], means[1],
                                      order=i)
    std = np.sqrt(np.diagonal(cov))
    label = r'{0}$(p={1:.3f})$'.format(label, p[0])
    funcs = get_polynomial_funcs(i)
    y = eval_result(x, p_opt, funcs)
    y_up = eval_result(x, p_opt + np.sign(p_opt) * std, funcs)
    y_down = eval_result(x, p_opt - np.sign(p_opt) * std, funcs)
    ax.plot(x, y, label=label, color=color)
    ax.plot(x, y_up, linestyle='--', color=color)
    ax.plot(x, y_down, linestyle='--', color=color)

ax.set(xlabel=r'Year',
       ylabel=fr'Mean Temperature [°C]',
       title=fr'Mean Temperature per Year')
ax.grid()
ax.legend()

# Find the peaks and estimate yearly cycle period
peaks = signal.find_peaks(df['Temperature'], distance=350)[0]

_, ax = plt.subplots()
ax.plot(df.index, df['Temperature'])
for peak in peaks:
    plt.axvline(x=peak, color='k', linestyle='--', linewidth=1)

ax.set(xlabel=r'Days',
       ylabel=fr'Temperature [°C]',
       title=fr'Peaks in Temperature Data')
ax.grid()

_, ax = plt.subplots()
period = peaks[1:] - peaks[:-1]
# Remove outlier
period = period[period < 380]
x = np.array(list(range(len(period))))
sigma = 10
ax.errorbar(x, period, linestyle='none',
            marker='x', yerr=sigma, capsize=5)
p_opt, p, cov, _ = polynomial_fit(x, period, sigma, order=1)
std = np.sqrt(np.diagonal(cov))
funcs = get_polynomial_funcs(1)
y = eval_result(x, p_opt, funcs)
y_up = eval_result(x, p_opt + np.sign(p_opt) * std, funcs)
y_down = eval_result(x, p_opt - np.sign(p_opt) * std, funcs)
ax.plot(x, y, label=r'Linear Fit $(p={0:.3f})$'.format(p[0]),
        color='tab:orange')
ax.plot(x, y_up, linestyle='--', color='tab:orange')
ax.plot(x, y_down, linestyle='--', color='tab:orange')

ax.set_ylim([320, 400])

ax.set(xlabel=fr'Index $i$',
       ylabel=fr'Period [Days]',
       title=fr'Peaks in Temperature Data')
ax.grid()
ax.legend()
plt.show()
