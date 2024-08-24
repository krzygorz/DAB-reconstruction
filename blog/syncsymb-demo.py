import numpy as np
import matplotlib.pyplot as plt
from dab_parameters import ref_sync_symbol, ref_sync_carriers

N_frame = 196608
N_null = 2656
fs = 2048000
N_guard = 504

#samples = np.fromfile('dab_short.iq', np.complex64)
samples = np.fromfile('DAB+_20240712_105527Z_2048000_176640000_float32_iq.raw', np.complex64)

samples = samples[:N_frame]
plt.plot(abs(samples))

plt.style.use('grayscale')

power = np.correlate(np.abs(samples)**2, np.ones(N_null)/(N_null), mode='valid')
null_start = np.argmin(power)

plt.figure(figsize=(8, 4), dpi=200)

freq_offset = -680
samples *= np.exp(2.0j*np.pi*np.arange(len(samples))/fs * freq_offset)

fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
xc = np.abs(np.correlate(samples, ref_sync_symbol, mode='valid'))
time_offset = np.argmax(xc)
ax.plot(xc)
plt.xlabel("correlation start [samples]")
plt.ylabel("correlation magnitude")
plt.margins(0)
#fig.savefig("sync-xc.svg")
plt.show()

print(null_start, time_offset, time_offset-N_null-N_guard)
