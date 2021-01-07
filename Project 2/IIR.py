import matplotlib.pyplot as plt

from library import (get_elliptic_filter, plot_filter, load_data,
                     apply_filter, plot_series, plot_spectrum)

# Load Data
path = 'Temperature_dataset.csv'
df = load_data(path)

# Design Filter
N = 4
rp = 20
rs = 40
Wn = 2.1
b, a = get_elliptic_filter(N, rp, rs, Wn)

df['Filtered'] = apply_filter(df['Temperature'], b, a)

plot_filter(b, a, Wn)
plot_series(df, ['Temperature', 'Filtered'],
                 ['Raw Data', 'Elliptic Filter'])
plot_spectrum(df, ['Temperature', 'Filtered'], ['Raw Data', 'Filtered'])

plt.show()
