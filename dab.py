import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift

from reference import ref_sync_symbol, ref_sync_carriers

fs = 2048000
N_carriers = 1536 
N_frame = 196608
N_null = 2656
N_symbol = 2552
N_guard = 504
N_fft = N_symbol-N_guard
symbols_per_frame = 76
carrier_spacing = fs/N_fft #1kHz

carrier_colors = np.zeros(N_fft)
carrier_colors[:N_carriers//2] = np.arange(0,N_carriers//2)
carrier_colors[-N_carriers//2+1:] = np.arange(N_carriers//2+1, N_carriers)

def find_null_offset(samples):
    power_est = np.correlate(np.abs(samples)**2, np.ones(N_null)/(N_null), mode='valid')
    null_start = np.argmin(power_est)
    return null_start

coarse_time_sync_margin = N_null//2
max_freq_offset = 16e3
def find_preamble_offset(samples):
    df = np.arange(-max_freq_offset,max_freq_offset,carrier_spacing)
    #correlations = np.zeros((len(df), len(samples)-N_fft+1),dtype='complex128')
    best_peak_mag = 0
    best_peak_idx = None
    #for n, freq_offset in enumerate(df):
    #    freq_corrected = samples * np.exp(-2.0j*np.pi*np.arange(len(samples))/fs * freq_offset)
    #    xc = np.correlate(freq_corrected, ref_sync_symbol, mode='valid')
    #    peak_idx = np.argmax(xc)
    #    peak_idx
    #    #correlations[n,:] = xc
    #    print(freq_offset)

    #plt.imshow(np.abs(correlations), interpolation='none')
    #plt.gca().set_aspect('auto')
    #plt.show()
    xc = np.correlate(samples, ref_sync_symbol, mode='valid')
    sync_symbol_start = np.argmax(np.abs(xc)) - N_guard
    return sync_symbol_start

def frame_sync(samples):
    null_symbol_start = find_null_offset(samples)
    null_symbol_end = null_symbol_start + N_null
    xcorr_start = null_symbol_end - coarse_time_sync_margin
    xcorr_end   = null_symbol_end + N_symbol + coarse_time_sync_margin

    sync_symbol_start = find_preamble_offset(samples[:N_frame])

    frame_start = sync_symbol_start-N_null
    frame_end = frame_start+N_frame
    frame = samples[frame_start:frame_end]
    remaining = samples[frame_end:]
    return frame, remaining

def simulate_channel(samples, normalized_time_offset=0, freq_offset_hz=0):
    N = len(samples)
    distorted = samples * np.exp(2.0j*np.pi*np.arange(N)/fs * freq_offset_hz)
    distorted = ifft(fft(distorted) * np.exp(2.0j*np.pi*np.arange(N)/N * normalized_time_offset))
    return distorted

def fine_freq_corr(recv_sync_symbol):
    offsets = recv_sync_symbol[:N_guard] * np.conj(recv_sync_symbol[N_fft:])
    avg_jump = np.angle(np.mean(offsets))/(N_fft)
    return avg_jump

class DQPSK_demod:
    def __init__(self, sync_symbol):
        self.prev_carriers = fft(sync_symbol[N_guard:])
    def __call__(self, symbol):
        carriers = fft(symbol[N_guard:])
        diff = self.prev_carriers * np.conj(carriers)
        self.prev_carriers = carriers
        # return np.r_[diff[-N_carriers//2:] diff[-N_carriers//2+1:]]
        return diff

def normalize_power(x):
    return x / np.sqrt(np.mean(np.abs(x)**2))

#remaining_samples = np.fromfile('DAB+_20240712_105527Z_2048000_176640000_float32_iq.raw', np.complex64)
#remaining_samples = np.fromfile('dab_short.iq', np.complex64)
from lofarsig import ref_resampled as remaining_samples

N = int(0.2*fs)
remaining_samples = remaining_samples[:N]
remaining_samples = normalize_power(remaining_samples)
#samples = simulate_channel(samples, freq_offset_hz=1000)

frame, remaining_samples = frame_sync(remaining_samples)

sync_symbol_start = N_null
sync_symbol_end = N_null+N_symbol
recv_sync_symbol = frame[sync_symbol_start:sync_symbol_end]
#recv_sync_fft_before_afc = fft(recv_sync_symbol[N_guard:])

afc_norm = fine_freq_corr(recv_sync_symbol)
print(f"AFC {afc_norm/(2*np.pi) * fs} Hz")

frame *= np.exp(1.0j * np.arange(N_frame) * afc_norm)
remaining_samples *= np.exp(1.0j * (np.arange(len(remaining_samples)) + N_frame) * afc_norm)

recv_sync_symbol = frame[sync_symbol_start:sync_symbol_end]

#recv_sync_fft_after_afc = fft(recv_sync_symbol[N_guard:])
#fig, ax = plt.subplots(1,2)
#ax[0].scatter(recv_sync_fft_before_afc.real, recv_sync_fft_before_afc.imag, c=carrier_colors)
#ax[0].set_title("before fine AFC")
#ax[1].scatter(recv_sync_fft_after_afc.real, recv_sync_fft_after_afc.imag, c=carrier_colors)
#ax[1].set_title("after fine AFC")

demod = DQPSK_demod(recv_sync_symbol)
data_symbols = frame[sync_symbol_end:]
data_symbols = np.split(data_symbols, symbols_per_frame-1)

soft = np.zeros((symbols_per_frame-1, N_fft), dtype="complex128")
for n, symbol in enumerate(data_symbols):
    data_diff = demod(symbol)
    soft[n,:] = data_diff
    if n==0:
        print(symbol)
        print(recv_sync_symbol)
#exit()

plt.figure()
for k,c in [(0,'red'), (1,'blue'), (100,'green'), (500,'yellow'), (2048-100,'orange')]:
    plt.scatter(soft[:,k].real, soft[:,k].imag, color=c, label=f"k={k}")
#plt.scatter(soft[:,0].real, soft[:,0].imag, color='red', label='first carrier')
#plt.scatter(soft[:,1].real, soft[:,1].imag, color='blue', label='second carrier')
#plt.scatter(soft[:,100].real, soft[:,100].imag, color='green', label='100')
plt.title("data differential constellation")
plt.legend()

#plt.figure()
#plt.hist(np.angle(soft.flatten()),256)

plt.show()
