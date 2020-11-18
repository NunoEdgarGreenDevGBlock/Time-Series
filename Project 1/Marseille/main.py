import pandas as pd
import matplotlib.pyplot as plt
import scipy.fft as fft
import numpy as np
from library import empricial_auto_correlation

path = './Marseille_dataset_T_H_P.csv'

data_raw = pd.read_csv(path, header=12)
data = data_raw.copy()

# Clean up date format to a single column
data.insert(2, 'date', pd.to_datetime({'year': data_raw['YEAR'],
                                       'month': data_raw['MO'],
                                       'day': data_raw['DY']}
                                      ))
del data['YEAR']
del data['MO']
del data['DY']

# Rename column names to better names
data.rename(columns={'LAT': 'latitude', 'LON': 'longitude',
                            'T2M': 'temperature', 'PS': 'pressure',
                            'QV2M': 'humidity'}, inplace=True)

print('Date Describtion:', data['date'].describe())

data.set_index('date', inplace=True)
# List missing date values, i.e. gaps
full_date_range = pd.date_range(
    min(data.index), max(data.index), freq='1D')
missing_dates = full_date_range[~full_date_range.isin(data.index)]
print('Missing Dates:', missing_dates)


# FFT
fft_temp = fft.rfft(data['temperature'].values)
fftfreq_daily = fft.rfftfreq(data['temperature'].size, 1)
fft_data = pd.DataFrame(data={'fft raw': fft_temp}, index=fftfreq_daily)
fft_data['real'] = np.real(fft_data['fft raw'])
fft_data['imag'] = np.imag(fft_data['fft raw'])
fft_data['psd'] = np.abs(fft_data['fft raw'])
fft_data['phase'] = np.arctan2(fft_data['real'], fft_data['imag'])

# Clean Data -> keep biggest N peaks
N = 5
cut_off = fft_data['psd'].sort_values().values[-N]
fft_data['cleaned'] = fft_data['fft raw']
clean_mask = fft_data['psd'].values < cut_off
fft_data.loc[clean_mask, 'cleaned'] = 0
data['ifft'] = fft.irfft(fft_data['cleaned'].values)

# Large DC Peak
fft_data['DC Peak'] = fft_data['fft raw']
fft_data['DC Peak'].iloc[1:] = 0
data['DC Peak'] = fft.irfft(fft_data['DC Peak'].values)

# Autocorrelation
rho = []
for t in range(1, len(data['temperature'])):
    rho.append(empricial_auto_correlation(data['temperature'].values, t))


std = np.sqrt((1 + 2 * abs(np.nansum(rho)) )/
              len(data['temperature'])) * np.std(data['temperature'])
print(np.std(data['temperature']))
print(
    'Avg. Temp.: ({0:.5f} +- {1:.5f})'.format(data['temperature'].mean(), std))

fig, ax = plt.subplots(2, 1)
ax[0].plot(fft_data.index, fft_data['psd'], color='tab:blue')
ax[0].set_xlabel('Frequency [1/Day]')
ax[0].set_ylabel('Amplitude')
ax[1].plot(fft_data.index, fft_data['phase'], color='tab:blue')
ax[1].set_xlabel('Frequency [1/Day]')
ax[1].set_ylabel('Phase')

# Plot
fig, ax = plt.subplots()
ax.plot(data.index, data['temperature'])
ax.set_xlabel('Date [YYYY]')
ax.set_ylabel('Temperature at 2 Meters [°C]')
ax2 = ax.twinx()
ax2.plot(data.index, data['ifft'], color='tab:orange')
ax2.set_ylabel('Inverse FFT with Noise Reduction [°C]', color='tab:orange')

# Plot
fig, ax = plt.subplots()
ax.plot(range(1,len(data['temperature'])), rho)
ax.set_xlabel('t [Day]')
ax.set_ylabel('Empricial Autocorrelation')

#pd.plotting.autocorrelation_plot(data['temperature'])

plt.show()


# # OLD
# psd = np.abs(fft)**2
# fftfreq_daily = fft.fftfreq(len(psd), 1)
# fftfreq_year = fft.fftfreq(len(psd), 1. / 365)
# # ax[1].plot(fftfreq_daily, 10 * np.log10(psd), label='Daily')
# ax[1].plot(fftfreq_year[fftfreq_year > 0], 10 *
#            np.log10(psd[fftfreq_year > 0]), label='Yearly')
# ax[1].set_xlabel('Date [YYYY]')
# ax[1].set_ylabel('Temperature at 2 Meters [°C]')

# plt.show()
