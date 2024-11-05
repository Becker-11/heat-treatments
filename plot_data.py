import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load the CSV file
df = pd.read_csv('Data/NL324.csv')
concentraion = df['concentration'].values / 10000
distance = df['distance'].values
# print(concentraion)
# print(distance)

plt.plot(distance, concentraion)
plt.show()