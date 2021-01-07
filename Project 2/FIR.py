import numpy as np
import matplotlib.pyplot as plt

from library import apply_boxcar, load_data

path = 'Temperature_dataset.csv'
df = load_data(path)

# Raw Data
_, ax = plt.subplots(figsize=(9.0, 2.5))
ax.plot(df['Date'], df['Temperature'], linewidth=0.75)
ax.set(xlabel=r'Date'.format(df['Date'][0].date()),
       ylabel=fr'Temperature [°C]')

# Boxcar
df['Boxcar'] = apply_boxcar(df['Temperature'], 15)
ax.plot(df['Date'], df['Boxcar'], linewidth=0.75)
ax.set(xlabel=r'Date'.format(df['Date'][0].date()),
       ylabel=fr'Temperature [°C]')

ax.grid()
plt.tight_layout()
plt.show()
