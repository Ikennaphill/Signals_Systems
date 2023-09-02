import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define the system parameters
K = 2.0
tau = 0.5

# Define the triangle wave input signal
T = 0.25
t_end = 1.75
t = np.arange(0, t_end, 0.01)
u = np.mod(t, T)/T
u[u < 0.5] = 2*u[u < 0.5]
u[u >= 0.5] = 2 - 2*u[u >= 0.5]

# Define the system transfer function
num = [K]
den = [tau, 1]

# Simulate the system response to the input signal
sys = signal.lti(num, den)
t_out, y_out, _ = signal.lsim2(sys, u, t)

# Plot the input and output signals
plt.plot(t, u, label='Input')
plt.plot(t_out, y_out, label='Output')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.title('System response to triangle wave input')
plt.legend()
plt.show()
