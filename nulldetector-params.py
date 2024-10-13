import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift
#from dab_parameters import (
#    fs, N_carriers, N_frame, N_symbol, N_null, ref_sync_symbol,
#    symbols_per_frame, zero_carrier_idx, carrier_indices
#)
N_frame = 196608
N_null = 2656

# samples = np.fromfile('data/DAB+_20240712_105527Z_2048000_176640000_float32_iq.raw', np.complex64)
samples = np.fromfile('data/dab_short.iq', np.complex64)[N_frame//2:]
signal_power = np.mean(np.abs(samples)**2)
snr = 2 # linear scale
N = len(samples)
noise = 1/np.sqrt(2) * np.sqrt(signal_power/snr) * (np.random.normal(size=N) + 1.0j*np.random.normal(size=N))
samples += noise

fig, ax = plt.subplots()
# fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
for segment in np.split(samples, np.arange(N_frame, len(samples), N_frame)):
    power = np.correlate(np.abs(segment)**2, np.ones(N_null)/(N_null), mode='valid')
    null_idx = np.argmin(power)
    
    delta = int(N_null*2)
    assert(null_idx > delta)
    segment_power = np.mean(np.abs(segment)**2)
    metric = power/segment_power
    print(null_idx)

    ax.plot(np.arange(-delta,delta), metric[null_idx-delta:null_idx+delta])#, color='black')
ax.set_xlabel("window start [samples]")
ax.set_ylabel("power inside window")
plt.ylim(0,1.1)
# plt.axhline(0.1)
ax.autoscale(enable=True, axis='x', tight=True)
# plt.savefig("power-multiple-runs.svg")
plt.show()
