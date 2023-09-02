#Imports
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

h = [3, 1, 0, -1] #Hypothetical impulse response
#h = [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0]
n = np.arange(0, len(h) , 1) #Indices of h
x = [-1, 2, 0, 1] #Input signal
#x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
m = np.arange(0, len(x) , 1) #Indices of x

#Perform convolution of x*h
convoluted = signal.convolve(x, h, mode='full')
nf = np.arange(0, len(convoluted), 1)
print('The colnvolution of x*h is ', convoluted.round(2))

#Plot the signals
fig, (ax_orig, ax_win, ax_conv) = plt.subplots(3, 1, sharex=True)
ax_orig.stem(m, x, use_line_collection=True)
ax_orig.set_title('Original pulse')
ax_orig.set_xlabel('n')
ax_orig.set_ylabel('x[n]')

ax_win.stem(n, h, use_line_collection=True)
ax_win.set_title('Impulse Response')
ax_win.set_xlabel('n')
ax_win.set_ylabel('h[n]')

ax_conv.stem(nf, convoluted, use_line_collection=True)
ax_conv.set_title('Convoluted Signal')
ax_conv.set_xlabel('n')
ax_conv.set_ylabel('y[n]')

fig.tight_layout()
plt.show()