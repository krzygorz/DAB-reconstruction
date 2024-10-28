import numpy as np
import scipy
from scipy import signal

fs_lofar = 200e6
fs_beamlet = fs_lofar/1024
fs_dab = 2048000

def beam_center_freq(mode,beamlet1, beamlet2):
    if mode==7:
        lower_edge = 200e6
    else:
        raise Exception(f"mode {mode} not implemented")
    return lower_edge + (beamlet2+beamlet1)/2 * fs_beamlet

def load_lofar_signal(mat_file_path, n_beamlets, variable='ref_signal'):
    print("loading reference signal...")
    mat_contents = scipy.io.loadmat(mat_file_path)
    ref_original = mat_contents[variable].flatten()
    print("resampling ...")
    # 2048000/(200e6/1024) = 2^15/5^5
    ref_resampled  = signal.resample_poly(ref_original, 2**15, 5**5 * n_beamlets)
    print("done")
    return ref_resampled

# 2048 / (200000/1024 * 10) = 2^14 / 5^6 =
# = (2^7 / 5^3) * (2^7) / (5^2)

#print("resampling [step 1]")
#tmp  = signal.resample_poly(ref_original, 2**7, 5**3)
#print("resampling [step 2]")
#ref_resampled = signal.resample_poly(tmp, 2**7, 5**2)
#print("done")

def save_lofar_signal(out_path, sig, n_beamlets, variable='reconstructed'):
    out_resampled = signal.resample_poly(sig, 5**5*n_beamlets, 2**15)
    scipy.io.savemat(out_path, {variable: out_resampled})
    # np.save("data/reconstructed_PIA.npy", sig)
