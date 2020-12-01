import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as tsa
import scipy.signal as signal
from fitting import (polynomial_fit, get_polynomial_funcs,
                     eval_result)
from library import (remove_trend, autocorrelation, noise, corrected_mean,
                     no_noise)
from outliers import smirnov_grubbs as grubbs

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

print(df.describe())
print(np.mean(df['Temperature']), np.std(df['Temperature']))
# Plot Raw Data
_, ax = plt.subplots(figsize=(9.0, 2.5))
ax.plot(df['Date'], df['Temperature'], linewidth=0.75)
ax.set(xlabel=r'Date'.format(df['Date'][0].date()),
       ylabel=fr'Temperature [°C]')

plt.tight_layout()
ax.grid()
plt.savefig('raw.png', dpi=300)

#
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

#Autocorrelation
df['Rho Own'] = autocorrelation(df['Temperature'])
df['Rho Stats'] = tsa.acf(df['Temperature'], nlags=len(df.index),
                          fft=False, missing='conservative')
df['Rho Stats Residual'] = tsa.acf(df['Residual Temperature'], nlags=len(
    df.index), fft=False, missing='conservative')
df['Rho Stats Stationary'] = tsa.acf(df['Stationary Temperature'], nlags=len(
    df.index), fft=False, missing='conservative')
alpha_1 = 1 / np.sqrt(len(df.index)) * np.ones(df.index.shape)
alpha_3 = 3 / np.sqrt(len(df.index)) * np.ones(df.index.shape)
temp = df['Rho Stats Residual']
percent_1 = 100 * temp[abs(temp) < alpha_1].count() / temp.count()
percent_3 = 100 * temp[abs(temp) < alpha_3].count() / temp.count()

_, ax = plt.subplots(figsize=(9.0, 6.0))
ax.plot(df.index, df['Rho Stats Stationary'],
        label='Autocorrelation Stationary')
ax.plot(df.index, df['Rho Stats Residual'], label='Autocorrelation Residual')
# ax.plot(df.index, df['Rho Stats'], label='Stats Autocorrelation',
#         marker='o', markersize=2)
# ax.plot(df.index, df['Rho Stats Stat'], label='Stats Autocorrelation Stat')
ax.plot(df.index, alpha_1, color='k', linewidth=0.5,
        label=r'$1\sigma$ Confidence (E.: $66.3~\%$; R.: ${'
              r':.1f}~\%$)'.format(
            percent_1))
ax.plot(df.index, -alpha_1, color='k', linewidth=0.5)
ax.plot(df.index, alpha_3, color='k', linestyle='--', linewidth=0.5,
        label=r'$3\sigma$ Confidence (E.: $99.7~\%$; R.: ${'
              r':.1f}~\%$)'.format(percent_3))
ax.plot(df.index, -alpha_3, color='k', linestyle='--', linewidth=0.5)
ax.set(xlabel=r'Lag $k$ [days]',
       ylabel=fr'Autocorrelation $\rho$ [1]',
       title=fr'Autocorrelation')
ax.grid()
ax.legend()
plt.tight_layout()
#
# plt.savefig('autocorrelation.png', dpi=1200)

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

_, ax = plt.subplots(figsize=(9.0, 5.0))

# Exponential fit converges to linear fit
labels = ['Linear Fit', 'Quadratic Fit', 'Cubic Fit']
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, label, color in zip(range(1, 4), labels, colors):
    p_opt, p, cov, _ = polynomial_fit(x, means[0], means[1],
                                      order=i)
    std = np.sqrt(np.diagonal(cov))
    label = r'{0} $(p={1:.3f})$'.format(label, p[0])
    funcs = get_polynomial_funcs(i)
    y = eval_result(x, p_opt, funcs)
    y_up = eval_result(x, p_opt + np.sign(p_opt) * std, funcs)
    y_down = eval_result(x, p_opt - np.sign(p_opt) * std, funcs)
    ax.plot(x, y, label=label, color=color, linewidth=1.5)
    if i == 3:
        ax.plot(x, y_up, linestyle='--', color=color, linewidth=1.5)
        ax.plot(x, y_down, linestyle='--', color=color, linewidth=1.5)
        ax.text(1.1, 27.10, r'$y=({3:.4f}\pm{7:.4f})*x^3+({2:.3f}\pm{'
                            r'6:.3f})*x^2+({1:.2f}\pm{5:.2f})*x+({0:.1f}\pm{'
                            r'4:.1f})$'.format(*p_opt, *std), fontsize=12,
                color=color)

ax.set(xlabel=r'Year Index [1]',
       ylabel=fr'Mean Temperature [°C]',
       title=fr'Mean Temperature per Year')

ax.errorbar(x, means[0], yerr=means[1], linestyle='none',
            capsize=5, marker='x', color='k', label='Data (naïve & improved)',
            linewidth=1.5)
ax.errorbar(x, means[0], yerr=means[2], linestyle='none',
            capsize=5, marker='x', color='k', linewidth=1.5)

plt.xticks(list(range(1, len(x) + 1, 2)))
ax.grid()
ax.legend(loc='lower right')
plt.tight_layout()
# plt.savefig('mean_plot.png', dpi=600)


# Find the peaks and estimate yearly cycle period
peaks = signal.find_peaks(df['Temperature'], distance=350)[0]
peaks = np.array([309, 694, 1029, 1398, 1774, 2146, 2502, 2857, 3239, 3587,
                  3951, 4287, 4698, 5038, 5401, 5774, 6139, 6516, 6878, 7256,
                  7598])
sigmas = 3+np.array([1, 2, 3, 3, 5, 7, 2, 2, 6, 7, 2, 2, 2, 3, 3, 1, 4, 32, 1,
                   3, 3])
_, ax = plt.subplots()
ax.plot(df.index, df['Temperature'], color='k', linewidth=0.5)
for sigma, peak in zip(sigmas, peaks):
    plt.axvline(x=peak, color='r', linestyle='-', linewidth=0.5)
    plt.axvline(x=peak - sigma, color='r', linestyle='--', linewidth=0.5)
    plt.axvline(x=peak + sigma, color='r', linestyle='--', linewidth=0.5)

ax.set(xlabel=r'Days',
       ylabel=fr'Temperature [°C]',
       title=fr'Peaks in Temperature Data')
ax.grid()

_, ax = plt.subplots(figsize=(9.0, 5.5))
period = peaks[1:] - peaks[:-1]
sigma_period = sigmas[1:] + sigmas[:-1]


x = np.array(list(range(1, len(period) + 1)))
ax.errorbar(x, period, linestyle='none',
            marker='x', yerr=sigma_period, capsize=5)


p_opt, p, cov, _ = polynomial_fit(x, period, sigma_period, order=3)
std = np.sqrt(np.diagonal(cov))
funcs = get_polynomial_funcs(3)
y = eval_result(x, p_opt, funcs)
y_up = eval_result(x, p_opt + std, funcs)
y_down = eval_result(x, p_opt - std, funcs)
ax.plot(x, y, label=r'Linear Fit $(p={0:.3f})$'.format(p[0]),
        color='tab:orange')
ax.plot(x, y_up, linestyle='--', color='tab:orange')
ax.plot(x, y_down, linestyle='--', color='tab:orange')

ax.set_ylim([320, 400])
plt.xticks(list(range(1, len(x) + 1, 2)))
ax.set(xlabel=fr'Index $i$ [1]',
       ylabel=fr'Period [days]',
       title=fr'Period between Temperature Peaks')
ax.grid()

plt.tight_layout()
plt.savefig('period.png', dpi=600)

print(p_opt, std)
print(np.mean(period), np.std(period) / np.sqrt(len(period)))
#

# Autocorrelation Matrix
years = np.array(list(range(3, 8)))
vector_stat = df['Stationary Temperature'].values[
              years[0] * 365:years[-1] * 365]
vector_noise = df['Residual Temperature'].values[
               years[0] * 365:years[-1] * 365]
autocorr_stat_matrix = np.outer(vector_stat, vector_stat)
autocorr_noise_matrix = np.outer(vector_noise, vector_noise)

years = np.array(list(range(0, 4)))
vector_stat = df['Rho Stats Stationary'].values[
              years[0] * 365:years[-1] * 365]
vector_noise = df['Rho Stats Residual'].values[
               years[0] * 365:years[-1] * 365]
autocorr_stat_matrix = np.outer(vector_stat, vector_stat)
autocorr_noise_matrix = np.outer(vector_noise, vector_noise)

fig = plt.figure(figsize=(9.0, 5.0))
ax0 = fig.add_subplot(1, 2, 1)
im = ax0.imshow(autocorr_stat_matrix, cmap='seismic', vmin=-1, vmax=1)
plt.xticks((years - years[0]) * 365, years)
plt.yticks((years - years[0]) * 365, years)
ax0.grid(color='k', linestyle='--', linewidth=0.5)
fig.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.15)
ax0.set(ylabel=fr'Years [1]', xlabel=fr'Years [1]', title=fr'Stationary')

ax1 = fig.add_subplot(1, 2, 2)
im = ax1.imshow(autocorr_noise_matrix, cmap='seismic', vmin=-0.1, vmax=0.1)
plt.xticks((years - years[0]) * 365, years)
plt.yticks((years - years[0]) * 365, years)
ax1.grid(color='k', linestyle='--', linewidth=0.5)
plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.15)
ax1.set(xlabel=r'Years [1]', title=fr'Residual')
plt.tight_layout()
# plt.savefig('matrix.png', dpi=600)
plt.show()
