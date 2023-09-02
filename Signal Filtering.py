  
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import fftpack
from scipy import signal
import scipy.io.wavfile
from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
def signal_samples(t):
#simulated signal with pure sinusoidal components at 1 Hz and at 22 Hz, on top of a
#normal-distributed noise floor.
 return (2*np.sin(2*np.pi*t) + 3*np.sin(2*22*np.pi*t) +
2.0*np.random.randn(*np.shape(t)))
B = 30.0 #Maximum frequency that is expected the signal
f_s = 2*B #Sampling frequency
delta_f = 0.01 #Resolution of frequency spectrum
N = int(f_s/delta_f) #Number of samples
T = N/f_s #Sampling period
t = np.linspace(0,T,N) #Array of sample times
f_t = signal_samples(t) #Signal values
#Use the FFT to obtain the frequncy domain representation of the above signal
F = fftpack.fft(f_t)
f = fftpack.fftfreq(N, 1.0/f_s) #contains the frequencies corresponding to each frequency bin.
mask = np.where(f >= 0) #Extracts the frequencies of interest
#Code to compute time domain representation from a filtered frequency domain reperesentation
F_filtered = F*(abs(f) < 2) #Low pass filter to remove frequencies above 2Hz
f_t_filtered = fftpack.ifft(F_filtered)
#Code to use convolution for filtering
t = np.linspace(0, T, N)
f_t = signal_samples(t)
H = abs(f) < 2
h = fftpack.fftshift(fftpack.ifft(H))
f_t_filtered_conv = signal.convolve(f_t, h, mode="same")
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot2grid((2,2), (0,0))
ax.plot(f, H)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("Frequency filter")
ax.set_ylim(0, 1.5)
ax = plt.subplot2grid((2,2), (0,1))
ax.plot(t - t[-1]/2.0, h.real)
ax.set_xlim(0, 20)
ax.set_xlabel("time (s)")
ax.set_ylabel("convolution kernel")
ax = plt.subplot2grid((2,2), (1,0), colspan=2)
ax.plot(t, f_t, label="original", alpha=0.25)
ax.plot(t, f_t_filtered.real, 'r', lw=2, label='filtered in frequency domain')
ax.plot(t, f_t_filtered_conv.real, 'b-', lw=2, label='filtered with convolution')
ax.set_xlim(0, 20)
ax.set_xlabel("time (s)")
ax.set_ylabel("signal")
ax.legend(loc=2)
fig.tight_layout()
plt.show()