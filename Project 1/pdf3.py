import pandas as pd
import numpy as np
#Reunion Island (1st location)
df=pd.read_csv('temperature_dataset.csv')
import matplotlib.pyplot as plt
plt.figure(figsize=[20,2])
plt.title('Temperature Time serie - 1st location')
plt.xlabel('Days')
plt.ylabel('Temperature °C')
T=df['T2M']
plt.plot(np.arange(0,1000,1),T[0:1000])
plt.vlines(0,20,28)
plt.vlines(365,20,28)
plt.vlines(365*2,20,28)

plt.show()