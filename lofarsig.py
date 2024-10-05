import numpy as np
import scipy
from scipy import signal

print("loading reference signal...")
mat_contents = scipy.io.loadmat("data/Signals_13_49_34_PIA.mat")
ref_original = mat_contents['ref_signal'].flatten()
# ref_original = mat_contents['sur_signal_x'].flatten()

fs_beam = 200e6/1024 * 10
fs_dab = 2048000

print("resampling ...")
ref_resampled  = signal.resample_poly(ref_original, 2**14, 5**6)
print("done")

# 2048 / (200000/1024 * 10) = 2^14 / 5^6 =
# = (2^7 / 5^3) * (2^7) / (5^2)

#print("resampling [step 1]")
#tmp  = signal.resample_poly(ref_original, 2**7, 5**3)
#print("resampling [step 2]")
#ref_resampled = signal.resample_poly(tmp, 2**7, 5**2)
#print("done")

def save_output_mat(sig):
    out_resampled = signal.resample_poly(sig, 5**6, 2**14)
    scipy.io.savemat("data/reconstructed_PIA.mat", {"reconstructed": out_resampled})
    np.save("data/reconstructed_PIA.npy", sig)
