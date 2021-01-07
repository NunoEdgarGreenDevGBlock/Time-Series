import matplotlib.pyplot as plt

from library import *

# Load Data
path = 'Temperature_dataset.csv'
df = load_data(path)

# Design Filter
N = 4
rp = 1
rs = 40
Wn = 2.1

# elliptic, butterworth, russian (rp =^= 'I')
b, a = get_butterworth_filter(N, Wn)

df['Elliptic'] = apply_filter(df['Temperature'], b, a)

plot_filter(b, a, Wn)
plot_series(df, ['Temperature', 'Elliptic'],
            ['Raw Data', 'Elliptic Filter'])
plot_spectrum(df, ['Temperature', 'Elliptic'], ['Raw Data', 'Elliptic'])

plt.show()
