import numpy as np
import matplotlib.pyplot as plt
from dab_parameters import ref_sync_symbol, N_null, N_symbol

def normalize_power(x):
    return x / np.sqrt(np.mean(np.abs(x)**2))

data_dir = 'data/'
samples = np.fromfile(data_dir+'dab_short.iq', np.complex64)
recording_sync_symbol = samples[N_null:N_null+N_symbol]
recording_sync_symbol = normalize_power(recording_sync_symbol)
ref_sync_symbol = normalize_power(ref_sync_symbol)

plt.plot(recording_sync_symbol.real, label="recording")
plt.plot(ref_sync_symbol.real, label="python")
plt.legend()
plt.show()
