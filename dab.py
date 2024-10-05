import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift
from time import time

from demod import DabDemod, modulate_frame
from sync import freq_shift, SubsampleTimeSync
from plots import (
    plot_frame_diff_constellation,
    plot_phase_error_histogram,
    plot_freqsync_history,
    ladder_plot,
    plot_regen_error,
    plot_regen_histogram,
    plot_timesync
)
from dab_parameters import (
    fs, N_carriers, N_frame, N_symbol, N_null, ref_sync_symbol,
    symbols_per_frame, zero_carrier_idx, carrier_indices
)
from fir_filter import filter_taps

def delay(signal, delay, N_taps=21, mode='same'):
    # https://pysdr.org/content/sync.html
    n = np.arange(-N_taps//2, N_taps//2)
    h = np.sinc(n - delay)
    h *= np.hamming(N_taps)
    h /= np.sum(h)
    return np.convolve(signal, h, mode=mode)

def normalize_power(x):
    return x / np.sqrt(np.mean(np.abs(x)**2))

data_dir = 'data/'
# full_signal = np.fromfile(data_dir+'dab_short.iq', np.complex64)
full_signal = np.fromfile(data_dir+'DAB+_20240712_105527Z_2048000_176640000_float32_iq.raw', np.complex64)
# from lofarsig import ref_resampled as full_signal
# full_signal = np.load(data_dir+'reconstructed.npy')[N_frame:]

freq_offset_hz = 50300
# full_signal = delay(full_signal, 0.3)

# full_signal = np.convolve(full_signal,filter_taps)
full_signal = normalize_power(full_signal)
# np.clip(full_signal.real, -3, 3, out=full_signal.real)
# np.clip(full_signal.imag, -3, 3, out=full_signal.imag)

remaining_samples = full_signal

n_frames = len(remaining_samples)//N_frame - 1
# n_frames = 1
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

timesync = SubsampleTimeSync()
t_start = time()
recv_frames = []
regen_frames = []
for i in range(0,n_frames):
    soft1, hard1, remaining_samples = demod(remaining_samples)
    recv_frames.append(demod.last_frame)
    regen = modulate_frame(hard1)
    regen_frames.append(regen)
    for symbol_data_carriers in soft1:
       timesync(symbol_data_carriers)
t_stop = time()
signal_duration = n_frames*N_frame/fs
processing_duration = t_stop-t_start
print(f"processed {signal_duration:0.3f}s signal in {processing_duration:0.3f}s (x{signal_duration/processing_duration:0.2f} speed)")

# regen = np.concatenate(regen_frames)
# # recv = np.concatenate(recv_frames)
# recv = full_signal[demod.initial_offset:demod.initial_offset+len(regen)]
#
# regen = freq_shift(regen, -demod.freqsync.freq_offset)
#
# # plt.figure()
# # nf = 4096
# # ninterp = 64
# # f = abs(ifft(fft(recv[:nf])*np.conj(fft(regen[:nf])), nf*ninterp))
# # plt.plot(f)
# # m = np.argmax(f)
# # if m > nf*ninterp//2:
# #     m = m-nf*ninterp
# # m /= ninterp
# # print(m)
#
# r = np.vdot(regen,recv)/np.vdot(regen,regen)
# regen = r*regen#*1.5
#
# s = slice(N_frame*20,N_frame*21)
# plot_regen_error(recv[s], regen[s])
# plot_regen_histogram(recv, regen)
#
# signal_power = np.sum(np.abs(regen)**2)
# noise_power = np.sum(np.abs(recv-regen)**2)
# snr = signal_power/noise_power
# print(f"SNR = {10*np.log10(snr):0.2f} dB")
#
# plt.figure()
# err = recv-regen
# plt.hist2d(err.real, err.imag, bins=64)
#
# plt.figure()
# plt.psd(recv,  label="recv")
# plt.psd(regen, label="regen")
# plt.legend()
#
#
# #plt.show()
# #exit()
#
# regen = normalize_power(regen)
# regen = np.concatenate([np.zeros(demod.initial_offset), regen])
# from lofarsig import save_output_mat
# save_output_mat(regen)

plot_timesync(timesync)

plot_frame_diff_constellation(soft1)
plot_phase_error_histogram(soft1)
ladder_plot(soft1)
plot_freqsync_history(demod.freqsync)
plt.show()
