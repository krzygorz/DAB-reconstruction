import numpy as np
from numpy import pi
from numpy.fft import fft, ifft, fftshift
from dab_parameters import (
    N_symbol, N_carriers, N_null, N_frame, N_guard,
    symbols_per_frame, get_data_carriers, carrier_indices,
    insert_null_carriers, mod_ofdm_symbol,
    ref_sync_symbol, ref_sync_carriers
)
from sync import FreqSync, cold_frame_sync, warm_frame_sync, SubsampleTimeSync

class NotEnoughSamplesError(Exception):
    pass

# that's not how the bits are mapped in the standard
def QPSK_decide(x):
    return np.round((np.angle(x)-pi/4) / (pi/2))
def QPSK_mod(x):
    phase = x*(pi/2) + pi/4
    return np.exp(1.0j*phase)
 
class DiffDemod:
    def __init__(self, sync_carriers):
        self.prev_carriers = sync_carriers
    def __call__(self, carriers):
        # diff = self.prev_carriers * np.conj(carriers)
        diff = np.conj(self.prev_carriers) * carriers
        self.prev_carriers = carriers
        return diff
class DiffMod:
    def __init__(self, sync_carriers):
        self.prev_carriers = sync_carriers
    def __call__(self, carriers):
        diff = self.prev_carriers * carriers
        self.prev_carriers = diff
        return diff

def demod_frame(frame, freqsync):
    sync_symbol_start = N_null
    sync_symbol_end = sync_symbol_start + N_symbol
    recv_sync_symbol = frame[sync_symbol_start:sync_symbol_end]
    recv_sync_symbol = freqsync(recv_sync_symbol)
    recv_sync_carriers = get_data_carriers(recv_sync_symbol)

    #freqresponse = insert_null_carriers(recv_sync_carriers) * np.conj(ref_sync_carriers) 
    #plt.figure()
    #plt.plot(np.arange(-N_fft*2, N_fft*2), np.abs(fftshift(ifft(freqresponse, N_fft*4))))
    #plt.xlim(-100,100)
    #plt.title('channel impulse response')
    
    diff = DiffDemod(recv_sync_carriers)
    data_symbols = frame[sync_symbol_end:]
    data_symbols = np.split(data_symbols, symbols_per_frame-1)
    
    soft = np.zeros((symbols_per_frame-1, N_carriers), dtype="complex128")
    hard = np.zeros((symbols_per_frame-1, N_carriers), dtype="u8")
    
    for n, symbol in enumerate(data_symbols):
        symbol = freqsync(symbol)
        data_carriers = get_data_carriers(symbol)
        data_diff = diff(data_carriers)
        soft[n,:] = data_diff
        hard[n,:] = QPSK_decide(data_diff)
    return soft

class DabDemod:
    def __init__(self):
        self.tracking = False
        self.freqsync = None
    def __call__(self, remaining_samples):
        if not self.tracking:
           frame_start, coarse_freq_offset = cold_frame_sync(remaining_samples)
           frame_end = frame_start+N_frame
           frame = remaining_samples[frame_start:frame_end]
           remaining_samples = remaining_samples[frame_end:]
           self.freqsync = FreqSync(coarse_freq_offset)
           self.tracking = True
           self.initial_offset = frame_start
           print(f"coarse frequency offset {coarse_freq_offset} Hz")
        else:
           frame, remaining_samples = warm_frame_sync(
               self.last_frame, remaining_samples, self.freqsync
           )

        if len(frame) < N_frame:
            raise NotEnoughSamplesError

        self.last_frame = frame
        soft = demod_frame(frame, self.freqsync)
        hard = QPSK_decide(soft)
        return soft, hard, remaining_samples

def modulate_frame(hard):
    frame = np.zeros(N_frame, dtype="complex128")
    frame[N_null:N_null+N_symbol] = ref_sync_symbol
    current_idx = N_null+N_symbol
    diff = DiffMod(ref_sync_carriers)
    for demod_data_carriers in hard:
        complex_data_carriers = QPSK_mod(demod_data_carriers)
        carriers = insert_null_carriers(complex_data_carriers)
        diff_carriers = diff(carriers)
        symbol = mod_ofdm_symbol(diff_carriers)
        frame[current_idx:current_idx+N_symbol] = symbol
        current_idx += N_symbol
    return frame

