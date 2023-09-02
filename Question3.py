from scipy import fftpack
from scipy import signal
import scipy.io.wavfile
from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Load the dataset from CSV file
data = pd.read_csv('relay_feedback.csv', header=None, names=["time", "temperature"])
data.time = (pd.to_datetime(data.time.values, unit="s")).tz_localize('UTC').tz_convert('Europe/Stockholm')

# Get the number of data points and the time interval
N = data.shape[0]
dt = data.iloc[1, 0] - data.iloc[0, 0]

# Extract the data values from the DataFrame
temperature = data.iloc[:, 1].values

# Apply the Hann window function
window = signal.blackman(N)
Windowed_temperature = temperature * window

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(data.time, temperature, label="original")
ax.plot(data.time, Windowed_temperature, label="windowed")
ax.set_ylabel("temperature", fontsize=14)
ax.legend(loc=0)
plt.show()



# Perform the FFT
fft = np.fft.fft(temperature)
freq = np.fft.fftfreq(N, dt)
mask = freq > 0
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlim(0.000005, 0.00004)
ax.axvline(1./40, color="r", lw=0.5) #add a line at the frequency corresponding to 24hr
ax.axvline(2./40, color="r", lw=0.5) #add a line at the frequency corresponding to 12hr
ax.axvline(3./40, color="r", lw=0.5) #add a line at the frequency corresponding to 8r
ax.plot(freq[mask], np.log(abs(fft[mask])), lw=2)
ax.set_ylabel("$\log|F|$", fontsize=14)
ax.set_xlabel("frequency (Hz)", fontsize=14)
plt.show()

