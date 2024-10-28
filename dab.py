import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift
from time import time
from lofarsig import load_lofar_signal, save_lofar_signal

from demod import DabDemod, modulate_frame
from sync import freq_shift, SubsampleTimeSync
from coherentsync import DDTimeSync
from plots import (
    plot_frame_diff_constellation,
    plot_phase_error_histogram,
    plot_freqsync_history,
    ladder_plot,
    plot_regen_error,
    plot_regen_histogram,
    plot_regen_error_histogram,
    plot_regen_spectrum,
    plot_timesync,
)
from dab_parameters import (
    fs, N_carriers, N_frame, N_symbol, N_null,
    symbols_per_frame, zero_carrier_idx, carrier_indices
)
from dsp_utils import normalize_power, delay

data_dir = 'data/'
out_file = data_dir+"regen.mat"
n_beamlets = 10
full_signal = load_lofar_signal("data/Signals_13_49_34_OLS.mat", n_beamlets);
# full_signal = np.fromfile(data_dir+'dab_short.iq', np.complex64)
# full_signal = np.fromfile(data_dir+'DAB+_20240712_105527Z_2048000_176640000_float32_iq.raw', np.complex64)
# from lofarsig import ref_resampled as full_signal
# full_signal = np.load(data_dir+'reconstructed.npy')[N_frame:]

full_signal = normalize_power(full_signal)
remaining_samples = full_signal

# n_frames = len(remaining_samples)//N_frame - 1
n_frames = 10
print(f"processing {n_frames} frames")

fine_sync = False

demod = DabDemod()
timesync = DDTimeSync(max_offset=1, N_taps=15, N_filters=16)
t_start = time()
recv_frames = []
regen_frames = []
for i in range(0,n_frames):
    soft1, hard1, remaining_samples = demod(remaining_samples)
    recv_frames.append(demod.last_frame)
    regen = modulate_frame(hard1)
    symbols_recv = demod.last_frame[N_null:].reshape((symbols_per_frame,-1))
    if fine_sync:
        symbols_regen = regen[N_null:].reshape((symbols_per_frame,-1))
        regen_frames.append(np.zeros(N_null))
        for s_rec, s_reg in zip(symbols_recv, symbols_regen):
            regen_sync = timesync(s_rec, s_reg)
            regen_frames.append(regen_sync)
    else:
        regen_frames.append(regen)
t_stop = time()
signal_duration = n_frames*N_frame/fs
processing_duration = t_stop-t_start
print(f"processed {signal_duration:0.3f}s signal in {processing_duration:0.3f}s (x{signal_duration/processing_duration:0.2f} speed)")

regen = np.concatenate(regen_frames)
# recv = np.concatenate(recv_frames)
recv = full_signal[demod.initial_offset:demod.initial_offset+len(regen)]
if fine_sync:
    recv = recv[:-timesync.processing_delay()]
    regen = regen[timesync.processing_delay():]

regen = freq_shift(regen, -demod.freqsync.freq_offset)

# plt.figure()
# nf = 4096
# ninterp = 64
# f = abs(ifft(fft(recv[:nf])*np.conj(fft(regen[:nf])), nf*ninterp))
# plt.plot(f)
# m = np.argmax(f)
# if m > nf*ninterp//2:
#     m = m-nf*ninterp
# m /= ninterp
# print(m)
# regen = delay(regen,m)

r = np.vdot(regen,recv)/np.vdot(regen,regen)
regen = r*regen
# regen = normalize_power(regen)
# recv = normalize_power(recv)

s = slice(N_frame*1,N_frame*2)
plot_regen_error(recv[s], regen[s])
plot_regen_histogram(recv, regen)
# plot_regen_error_histogram(recv,regen)

signal_power = np.sum(np.abs(regen)**2)
noise_power = np.sum(np.abs(recv-regen)**2)
snr = signal_power/noise_power
print(f"SNR = {10*np.log10(snr):0.2f} dB")

plot_regen_spectrum(recv, regen)

regen = normalize_power(regen)
regen = np.concatenate([np.zeros(demod.initial_offset), regen])
save_lofar_signal(out_file, regen, n_beamlets)

if fine_sync:
    plot_timesync(timesync)

plot_frame_diff_constellation(soft1)
plot_phase_error_histogram(soft1)
ladder_plot(soft1)
plot_freqsync_history(demod.freqsync)
plt.show()
