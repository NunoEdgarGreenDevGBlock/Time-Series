import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from library import remove_trend, autocorrelation, noise, autocorrelation_pd
import statsmodels.tsa.stattools as tsa

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

print(df.head())

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
# ax.plot(df.index, df['Rho Own'], label='Autocorrelation')
ax.plot(df.index, df['Rho Stats Stationary'],
        label='Autocorrelation Stationary')
ax.plot(df.index, df['Rho Stats Residual'], label='Autocorrelation Residual')
# ax.plot(df.index, df['Rho Stats'], label='Stats Autocorrelation',
#         marker='o', markersize=2)
# ax.plot(df.index, df['Rho Stats Stat'], label='Stats Autocorrelation Stat')
ax.plot(df.index, alpha_005, color='k', linewidth=0.5)
ax.plot(df.index, -alpha_005, color='k', linewidth=0.5)
ax.plot(df.index, alpha_001, color='k', linestyle='--', linewidth=0.5)
ax.plot(df.index, -alpha_001, color='k', linestyle='--', linewidth=0.5)
ax.set(xlabel=r'Lag $k$ [days]',
       ylabel=fr'Autocorrelation $\rho$ [1]',
       title=fr'Autocorrelation')
ax.grid()
ax.legend()
plt.show()
