import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift
from time import time

from demod import DabDemod, modulate_frame
from sync import freq_shift
from plots import (
    plot_frame_diff_constellation,
    plot_phase_error_histogram,
    plot_freqsync_history,
    ladder_plot,
    plot_regen_error,
    plot_coherent_error
)
from dab_parameters import (
    fs, N_carriers, N_frame, N_symbol, N_null, ref_sync_symbol,
    symbols_per_frame, zero_carrier_idx, carrier_indices
)

def simulate_channel(samples, normalized_time_offset=0, freq_offset_hz=0):
    N = len(samples)
    distorted = freq_shift(samples, freq_offset_hz)
    distorted = ifft(fft(distorted) * np.exp(2.0j*pi*np.arange(N)/N * normalized_time_offset))
    return distorted

def normalize_power(x):
    return x / np.sqrt(np.mean(np.abs(x)**2))

data_dir = 'data/'
# full_signal = np.fromfile(data_dir+'dab_short.iq', np.complex64)
#full_signal = np.fromfile(data_dir+'DAB+_20240712_105527Z_2048000_176640000_float32_iq.raw', np.complex64)
from lofarsig import ref_resampled as full_signal
# full_signal = np.load(data_dir+'reconstructed.npy')[N_frame:]

# freq_offset_hz = 50300
# full_signal = simulate_channel(full_signal, freq_offset_hz=freq_offset_hz)
full_signal = normalize_power(full_signal)

remaining_samples = full_signal

n_frames = len(remaining_samples)//N_frame - 1
# n_frames = 4
print(f"processing {n_frames} frames")

demod = DabDemod()
# soft, hard, remaining_samples = demod(remaining_samples)

# demod = DabDemod()
# soft_f0, hard_f0, _ = demod(remaining_samples_unshifted)
# 
# plt.plot(np.angle(soft[0]), 'o', label="shift")
# plt.plot(np.angle(soft_f0[0]), '+', label="no shift")
# plt.legend()
# plt.show()
# exit()

# soft_sym = soft[0]
# # soft_sym = normalize_power(soft_sym)
# hard_sym = hard[0]
# from demod import QPSK_mod, mod_ofdm_symbol, DiffMod
# from dab_parameters import insert_null_carriers, ref_sync_carriers
# st = N_null + demod.initial_offset + N_symbol
# recv_symbol = full_signal[st:st+N_symbol*1]
# recv_symbol = normalize_power(recv_symbol)
# complex_data_carriers = QPSK_mod(hard_sym)
# 
# plt.figure()
# diff = DiffMod(ref_sync_carriers)
# 
# carriers = insert_null_carriers(complex_data_carriers)
# gen_symbol = mod_ofdm_symbol(diff(carriers))
# gen_symbol = normalize_power(gen_symbol)
# phase = np.angle(np.sum(np.conj(gen_symbol)*recv_symbol))
# gen_symbol *= np.exp(1.0j*phase)
# print(phase)
# # gen_symbol = freq_shift(gen_symbol, freq_offset_hz)
# # recv_symbol = freq_shift(recv_symbol, -freq_offset_hz)
# 
# plt.plot(abs(recv_symbol), 'o', label="recv")
# plt.plot(abs(gen_symbol), 'x', label="regen")
# plt.legend()
# 
# error = gen_symbol-recv_symbol
# plt.figure()
# plt.plot(abs(fft(error)), label="regen")
# plt.title("error spectrum")
# plt.figure()
# plt.title("error in time domain")
# plt.plot(abs(error),  label="regen")
# plt.figure()
# plt.title("phase error")
# plt.plot(np.angle(gen_symbol * np.conj(recv_symbol)),  label="regen")
# plt.show()
# exit()
#######################################

#timesync = SubsampleTimeSync()
t_start = time()
recv_frames = []
regen_frames = []
for i in range(0,n_frames):
    soft1, hard1, remaining_samples = demod(remaining_samples)
    recv_frames.append(demod.last_frame)
    regen = modulate_frame(hard1)
    regen_frames.append(regen)
#    for symbol_data_carriers in soft:
#        timesync(symbol_data_carriers)
t_stop = time()
signal_duration = n_frames*N_frame/fs
processing_duration = t_stop-t_start
print(f"processed {signal_duration:0.3f}s signal in {processing_duration:0.3f}s (x{signal_duration/processing_duration:0.2f} speed)")

regen = np.concatenate(regen_frames)
# recv = np.concatenate(recv_frames)
recv = full_signal[demod.initial_offset:demod.initial_offset+len(regen)]
print(len(recv), len(regen))

regen = freq_shift(regen, -demod.freqsync.freq_offset)
phase = np.angle(np.sum(np.conj(regen)*recv))
regen *= np.exp(1.0j*phase)
print(f"regen initial phase {phase:0.2f}")
regen = normalize_power(regen)

# plot_regen_error(recv, regen)
# plot_coherent_error(recv, regen)
# plt.show()

regen = np.concatenate([np.zeros(demod.initial_offset), regen])
from lofarsig import save_output_mat
regen = normalize_power(regen)
print(len(regen), len(full_signal))
save_output_mat(regen)

# plot_timesync(timesync)

plot_frame_diff_constellation(soft1)
plot_phase_error_histogram(soft1)
ladder_plot(soft1)
plot_freqsync_history(demod.freqsync)
plt.show()
