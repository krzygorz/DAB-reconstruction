import numpy as np
import scipy
from scipy import signal

fs_lofar = 200e6
fs_beamlet = fs_lofar/1024
fs_beam = fs_beamlet * 10
fs_dab = 2048000

def beam_center_freq(mode,beamlet1, beamlet2):
    if mode==7:
        lower_edge = 200e6
    else:
        raise Exception(f"mode {mode} not implemented")
    return lower_edge + (beamlet2+beamlet1)/2 * fs_beamlet

def load_lofar_signal(mat_file_path, variable='ref_signal'):
    print("loading reference signal...")
    mat_contents = scipy.io.loadmat(mat_file_path)
    ref_original = mat_contents[variable].flatten()
    print("resampling ...")
    ref_resampled  = signal.resample_poly(ref_original, 2**14, 5**6)
    print("done")
    return ref_resampled

# 2048 / (200000/1024 * 10) = 2^14 / 5^6 =
# = (2^7 / 5^3) * (2^7) / (5^2)

#print("resampling [step 1]")
#tmp  = signal.resample_poly(ref_original, 2**7, 5**3)
#print("resampling [step 2]")
#ref_resampled = signal.resample_poly(tmp, 2**7, 5**2)
#print("done")

def save_lofar_signal(out_path, sig, variable='reconstructed'):
    out_resampled = signal.resample_poly(sig, 5**6, 2**14)
    scipy.io.savemat(out_path, {variable: out_resampled})
    # np.save("data/reconstructed_PIA.npy", sig)
