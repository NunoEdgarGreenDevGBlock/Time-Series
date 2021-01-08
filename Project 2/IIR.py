import matplotlib.pyplot as plt

from library import *

# Load Data
path = 'Temperature_dataset.csv'
df = load_data(path)

# Design Filter
N = 8
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

# Notes: The Russian II filter seems a terrible idea. It places the cut off too
# soon. The type I is better but has ripples in the pass band and looking at the
# plots I think we can endure ripples in the stop band more easily. Thus I would
# tend towards the butterworth as it has no ripples in the pass band and the
# slope seems fine. For a steeper slope we can just crank up the order.
# I couldn't find a stability criterion other than "the poles get too close to
# the unit circle". I've read that it depends on the computer and numerical
# algorithm so we can't influence it. For me on my PC butterworth of 8th order
# seems fine, 9th has already artifacts and everything above is completely
# broke.
