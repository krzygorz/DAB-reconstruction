import numpy as np
from numpy import pi
from dab_parameters import ref_sync_symbol, ref_sync_carriers
from dab_parameters import fs, N_carriers, N_frame, N_null, N_symbol, N_guard, N_fft, symbols_per_frame, carrier_spacing
from numpy.fft import fft, ifft, fftshift

def find_null_offset(samples):
    power_est = np.correlate(np.abs(samples)**2, np.ones(N_null)/(N_null), mode='valid')
    null_start = np.argmin(power_est)
    return null_start

def freq_shift(samples, freq_offset):
    return samples * np.exp(2.0j*pi*np.arange(len(samples))/fs * freq_offset)

coarse_time_sync_margin = N_null//2
max_freq_offset = 200e3
def find_preamble_offset(samples):
    df = np.arange(-max_freq_offset,max_freq_offset+carrier_spacing,carrier_spacing)
    #correlations = np.zeros((len(df), len(samples)-N_fft+1))
    best_peak_mag = 0
    best_peak_idx = None
    best_peak_freq = None
    for n, freq_offset in enumerate(df):
        freq_corrected = freq_shift(samples, freq_offset)
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

    sync_symbol_start = best_peak_idx
    return sync_symbol_start, best_peak_freq

warm_margin = 20
def warm_frame_sync(prev_frame, samples, freqsync):
    xcorr_start = N_null - warm_margin
    xcorr_end = N_null + N_symbol + warm_margin
    xcorr_input = freqsync.correct(samples[xcorr_start:xcorr_end])
    xc = np.correlate(xcorr_input, ref_sync_symbol, mode='valid')
    sync_symbol_start = np.argmax(np.abs(xc)) + xcorr_start
    frame_start = sync_symbol_start - N_null
    frame_end = frame_start + N_frame
    if frame_start >= 0:
        frame = samples[frame_start:frame_end]
        if frame_start > 0:
            print("[frame sync] skipping 1 frame")
    elif frame_start < 0:
        frame = np.concatenate([prev_frame[frame_start:], samples[:frame_end]])
        print("[frame sync] going back 1 frame")
    remaining = samples[frame_end:]
    return frame, remaining

def cold_frame_sync(samples):
    null_symbol_start = find_null_offset(samples[:N_frame])
    null_symbol_end = null_symbol_start + N_null
    xcorr_start = null_symbol_end - coarse_time_sync_margin
    xcorr_end   = null_symbol_end + N_symbol + coarse_time_sync_margin

    sync_symbol_start, coarse_freq_offset = find_preamble_offset(samples[xcorr_start:xcorr_end])
    sync_symbol_start += xcorr_start
    #print(null_symbol_end)
    #print(sync_symbol_start)

    frame_start = sync_symbol_start-N_null
    return frame_start, coarse_freq_offset

def estimate_fine_freq_offset(symbol):
    offsets = symbol[:N_guard] * np.conj(symbol[N_fft:])
    avg_jump = np.angle(np.mean(offsets))/(N_fft)
    freq_offset = avg_jump/(2*pi) * fs
    #plt.figure()
    #plt.hist(np.angle(offsets)/N_fft /(2*pi) * fs, 200)
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
        self.history = []
        self.sum = 0
        self.n = 0
    def __call__(self, symbol):
        self.n += 1
        t = np.arange(0,N_symbol) + self.phase_counter
        corrected = symbol*np.exp(2.0j * pi * t * self.freq_offset / fs)

        extra_freq_offset = estimate_fine_freq_offset(corrected)
        if self.phase_counter == 0:
            self.freq_offset += extra_freq_offset
            self.sum = self.freq_offset
        else:
            # self.freq_offset += self.alpha*extra_freq_offset
            self.sum += self.freq_offset + extra_freq_offset
            self.freq_offset = self.sum / self.n
        self.history.append(self.freq_offset)

        symbol = symbol* np.exp(2.0j * pi * t * self.freq_offset / fs)

        self.phase_counter += N_symbol
        return symbol
    def correct(self, data):
        t = np.arange(0,len(data)) + self.phase_counter
        return data * np.exp(2.0j * pi * t * self.freq_offset / fs)

def mod_angle(x, maxangle=2*pi):
    return ((x+maxangle/2) % maxangle) - maxangle/2

class SubsampleTimeSync:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.history = []
        self.time_offset = 0

    def __call__(self, data_carriers):
        n_diff = 100
        angles = np.angle(data_carriers)
        diff = mod_angle(angles[n_diff:] - angles[:-n_diff], pi/2)
        self.time_offset += np.mean(diff)/n_diff
        self.history.append(self.time_offset / (2*pi) * N_fft)
    
        # plt.figure()
        # k = np.arange(N_fft)
        # k[-N_fft//2+1:] -= N_fft
        # plt.plot(k, k*time_offset)
        # plt.plot(carrier_indices,angles, 'o')
        # plt.axhline(-pi/2, color='black', linestyle='--', alpha=0.3)
        # plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        # plt.axhline(pi/2, color='black', linestyle='--', alpha=0.3)
        # plt.ylim(-pi,pi)
        # plt.show()
    
