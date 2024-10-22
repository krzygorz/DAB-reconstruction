import numpy as np
import matplotlib.pyplot as plt
from dab_parameters import ref_sync_symbol, ref_sync_carriers, N_null, N_symbol, N_fft, N_frame

def normalize_power(x):
    return x / np.sqrt(np.mean(np.abs(x)**2))

data_dir = 'data/'
samples = np.fromfile(data_dir+'dab_short.iq', np.complex64)
recording_sync_symbol = samples[N_null:N_null+N_symbol]

# recording_sync_symbol = normalize_power(recording_sync_symbol)
# ref_sync_symbol = normalize_power(ref_sync_symbol)
# plt.plot(recording_sync_symbol.real, label="recording")
# plt.plot(ref_sync_symbol.real, label="python")
# plt.legend()
# plt.show()

c_sym = np.fromfile("C/build/dump.txt", np.complex64)
py_sym = ref_sync_symbol
rec_sym = recording_sync_symbol

c_sym = normalize_power(c_sym)
rec_sym = normalize_power(rec_sym)
py_sym = normalize_power(py_sym)
# py_carriers *= N_fft
# rec_sym /= N_fft

plt.plot(c_sym.real, label="C")
# plt.plot(rec_sym.real, '-', label="recording")
#plt.plot(abs(py_sym - rec_sym))
plt.figure()
plt.plot(np.abs(np.correlate(samples[:N_frame], c_sym, mode='valid')), label='C')
plt.plot(np.abs(np.correlate(samples[:N_frame], py_sym, mode='valid')), label='py')
plt.figure()
plt.plot(np.abs(np.correlate(c_sym, py_sym, mode='valid')), label='C')
plt.legend()
plt.show()
