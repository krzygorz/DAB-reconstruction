import scipy
from scipy import signal

print("loading reference signal...")
mat_contents = scipy.io.loadmat("Signals_13_49_34_OLS.mat")
ref_original = mat_contents['ref_signal'].flatten()

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

