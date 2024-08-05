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
max_freq_offset = 90e3
def find_preamble_offset(samples):
    df = np.arange(-max_freq_offset,max_freq_offset+carrier_spacing,carrier_spacing)
    #correlations = np.zeros((len(df), len(samples)-N_fft+1))
    best_peak_mag = 0
    best_peak_idx = None
    best_peak_freq = None
    for n, freq_offset in enumerate(df):
        freq_corrected = samples * np.exp(2.0j*np.pi*np.arange(len(samples))/fs * freq_offset)
        xc = np.abs(np.correlate(freq_corrected, ref_sync_symbol, mode='valid'))
        #correlations[n,:] = xc

        peak_idx = np.argmax(xc)
        peak_mag = xc[peak_idx]
        if peak_mag > best_peak_mag:
            best_peak_mag = peak_mag
            best_peak_idx = peak_idx
            best_peak_freq = freq_offset

    #plt.figure()
    #plt.imshow(correlations, interpolation='none', extent=[0,correlations.shape[1],df[0],max_freq_offset])
    #plt.gca().set_aspect('auto')

    sync_symbol_start = best_peak_idx - N_guard
    return sync_symbol_start, best_peak_freq

    #xc = np.correlate(samples, ref_sync_symbol, mode='valid')
    #sync_symbol_start = np.argmax(np.abs(xc)) - N_guard
    #return sync_symbol_start, 0

zero_carrier_idx = N_carriers//2
def remove_guard(symbol):
    return symbol[N_guard:]
def get_data_carriers(symbol):
    all_carriers = fft(remove_guard(symbol))
    return np.r_[all_carriers[-N_carriers//2:], all_carriers[:N_carriers//2]]
def insert_null_carriers(data_carriers):
    negative = data_carriers[:N_carriers//2]
    positive = data_carriers[N_carriers//2:]
    carriers = np.zeros(N_fft, dtype=np.complex64)
    carriers[:N_carriers//2] = positive
    carriers[-N_carriers//2:] = negative
    return carriers
carrier_indices = np.arange(-N_carriers//2, N_carriers//2)

def frame_sync(samples):
    null_symbol_start = find_null_offset(samples)
    null_symbol_end = null_symbol_start + N_null
    xcorr_start = null_symbol_end - coarse_time_sync_margin
    xcorr_end   = null_symbol_end + N_symbol + coarse_time_sync_margin

    sync_symbol_start, coarse_freq_offset = find_preamble_offset(samples[xcorr_start:xcorr_end])
    sync_symbol_start += xcorr_start

    frame_start = sync_symbol_start-N_null
    frame_end = frame_start+N_frame
    frame = samples[frame_start:frame_end]
    remaining = samples[frame_end:]
    return frame, remaining, coarse_freq_offset

def simulate_channel(samples, normalized_time_offset=0, freq_offset_hz=0):
    N = len(samples)
    distorted = samples * np.exp(2.0j*np.pi*np.arange(N)/fs * freq_offset_hz)
    distorted = ifft(fft(distorted) * np.exp(2.0j*np.pi*np.arange(N)/N * normalized_time_offset))
    return distorted

def fine_freq_corr(recv_symbol):
    offsets = recv_symbol[:N_guard] * np.conj(recv_symbol[N_fft:])
    avg_jump = np.angle(np.mean(offsets))/(N_fft)
    freq_offset = avg_jump/(2*np.pi) * fs

    #plt.figure()
    #plt.hist(np.angle(offsets)/N_fft /(2*np.pi) * fs, 200)
    #plt.title("fine AFC histogram")
    #plt.xlabel("frequency offset [Hz]")
    #plt.ylabel("sample count")
    #plt.axvline(freq_offset, color='red', alpha=0.5, linewidth=1)
    #plt.show()

    return freq_offset

class FreqSync:
    def __init__(self, coarse_offset):
        self.freq_offset = coarse_offset
        self.phase_counter = 0
    def __call__(self, samples):
        t = np.arange(0,len(samples)) + phase_counter
        self.phase_counter += len(samples)
        return symbol * np.exp(2.0j * np.pi * t * self.freq_offset / fs)
 
class DQPSK_demod:
    def __init__(self, sync_carriers):
        self.prev_carriers = sync_carriers
        #self.last_time_offset = 0
        #self.n = 0
        #self.drift = np.zeros(symbols_per_frame-1)
    def __call__(self, carriers):
        diff = self.prev_carriers * np.conj(carriers)

        #time_offset = superfine_time_sync(diff)
        #diff *= np.exp(-1.0j * carrier_indices * time_offset)
        #self.last_time_offset += time_offset
        #self.drift[self.n] = time_offset
        #self.n += 1

        self.prev_carriers = carriers
        return diff

def mod_angle(x, maxangle=2*np.pi):
    return ((x+maxangle/2) % maxangle) - maxangle/2

def superfine_time_sync(data_carriers):
    n_diff = 100
    angles = np.angle(data_carriers)
    diff = mod_angle(angles[n_diff:] - angles[:-n_diff], np.pi/2)
    time_offset = np.mean(diff)/n_diff

    # plt.figure()
    # k = np.arange(N_fft)
    # k[-N_fft//2+1:] -= N_fft
    # plt.plot(k, k*time_offset)
    # plt.plot(carrier_indices,angles, 'o')
    # plt.axhline(-np.pi/2, color='black', linestyle='--', alpha=0.3)
    # plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    # plt.axhline(np.pi/2, color='black', linestyle='--', alpha=0.3)
    # plt.ylim(-np.pi,np.pi)
    # plt.show()

    return time_offset

def normalize_power(x):
    return x / np.sqrt(np.mean(np.abs(x)**2))

def demod_frame(frame):
    recv_sync_symbol = frame[sync_symbol_start:sync_symbol_end]
    recv_sync_carriers = get_data_carriers(recv_sync_symbol)

    plt.figure()
    plt.plot(np.arange(-N_fft*2, N_fft*2), np.abs(fftshift(ifft(insert_null_carriers(recv_sync_carriers) * np.conj(ref_sync_carriers), N_fft*4))))
    plt.xlim(-100,100)
    plt.title('channel impulse response')
    
    demod = DQPSK_demod(recv_sync_carriers)
    data_symbols = frame[sync_symbol_end:]
    data_symbols = np.split(data_symbols, symbols_per_frame-1)
    
    soft = np.zeros((symbols_per_frame-1, N_carriers), dtype="complex128")
    
    offsets = np.zeros(symbols_per_frame-1)
    offsets2 = np.zeros(symbols_per_frame-1)
    offset_sum = 0
    for n, symbol in enumerate(data_symbols):
        symbol_offset_est = fine_freq_corr(symbol)
        offset_sum += symbol_offset_est
        est_offset = offset_sum/(n+1)
    
        t = np.arange(0,N_symbol) + N_symbol*(n+1)
        symbol *= np.exp(2.0j * np.pi * t * est_offset / fs)
    
        offsets[n] = symbol_offset_est
        offsets2[n] = est_offset
    
        data_carriers = get_data_carriers(symbol)
        data_diff = demod(data_carriers)
        soft[n,:] = data_diff
    #plt.figure()
    #plt.plot(demod.drift)
    plt.figure()
    plt.plot(offsets, 'o')
    plt.plot(offsets2)

    return soft

###################################################

#remaining_samples = np.fromfile('dab_short.iq', np.complex64)
remaining_samples = np.fromfile('DAB+_20240712_105527Z_2048000_176640000_float32_iq.raw', np.complex64)
#from lofarsig import ref_resampled as remaining_samples

###################################################


N = int(0.2*fs)
remaining_samples = remaining_samples[:N]
remaining_samples = normalize_power(remaining_samples)
#samples = simulate_channel(samples, freq_offset_hz=1000)

frame, remaining_samples, coarse_freq_offset = frame_sync(remaining_samples)
print(f"coarse AFC {coarse_freq_offset} Hz")

sync_symbol_start = N_null
sync_symbol_end = N_null+N_symbol
recv_sync_symbol = frame[sync_symbol_start:sync_symbol_end]

fine_freq_offset = fine_freq_corr(recv_sync_symbol)
print(f"fine AFC {fine_freq_offset} Hz")

freq_offset = coarse_freq_offset + fine_freq_offset

t_frame = np.arange(N_frame)
t_remaining = np.arange(len(remaining_samples)) + N_frame
frame *= np.exp(2.0j * np.pi * t_frame * freq_offset / fs)
remaining_samples *= np.exp(2.0j * np.pi * t_remaining * freq_offset / fs)

soft = demod_frame(frame)

plt.figure()
k=np.arange(0,N_carriers,10)
iq_scatter = plt.scatter(soft[:,k].real, soft[:,k].imag, c=np.tile(k, symbols_per_frame-1))
cbar = plt.colorbar(iq_scatter)
cbar.set_label('carrier no.')
plt.title("data differential constellation")

from scipy.stats import norm
plt.figure()
without_zero = np.delete(soft, zero_carrier_idx, axis=1).flatten()
phase_err = np.angle(without_zero) % (np.pi/2) - np.pi/4
n_bins = 256
_, bins, _ = plt.hist(phase_err, n_bins, density=True, label="histogram", log=True)
mu,stddev = norm.fit(phase_err)
plt.ylim(1e-3,2*np.pi)
plt.xlim(-np.pi/4,np.pi/4)
plt.plot(bins,norm.pdf(bins,mu,stddev), label="normal distribution fit")
plt.title(f"phase error PDF [mu = {mu:0.4f} stddev = {stddev:0.4f}]")
plt.legend()

from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots()
num_frames, _ = soft.shape
y = np.angle(soft).T
x = np.broadcast_to(carrier_indices[:, np.newaxis], y.shape)
c = np.broadcast_to(np.arange(0,num_frames)[np.newaxis,:], y.shape)
scatterplot = ax.scatter(x,y, s=1, c=c, cmap='Reds')
ax.set_xlim(-N_carriers//2, N_carriers//2)
ax.set_ylim(-np.pi,np.pi)
ax.set_xlabel('carrier no')
ax.set_ylabel('phase')
ax.axhline(-np.pi/2, color='black', linestyle='--')
ax.axhline(0, color='black', linestyle='--')
ax.axhline(np.pi/2, color='black', linestyle='--')

#divider   = make_axes_locatable(ax)
#ax_stddev = divider.append_axes("top", 0.3, pad=0.1, sharex=ax)
#stddev =np.std(np.angle(soft)%(np.pi/2)-np.pi/4,axis=0) 
#ax_stddev.plot(carrier_indices, stddev)
#ax_stddev.set_ylim(0, np.max(np.delete(stddev,zero_carrier_idx)))
#ax_stddev.xaxis.set_tick_params(labelbottom=False)

#ax_cbar = divider.append_axes("right", 0.2, pad=0.1)
#cbar = fig.colorbar(scatterplot, cax=ax_cbar)
cbar = fig.colorbar(scatterplot)
cbar.set_label('frame no.')

plt.show()
