import numpy as np
from numpy import pi
from numpy.fft import fft, ifft, fftshift
from scipy import signal

def fir_delay(signal, delay, N_taps=21):
    # https://pysdr.org/content/sync.html
    n = np.arange(-N_taps//2, N_taps//2)+1
    h = np.sinc(n - delay)
    h *= np.hamming(N_taps)
    h /= np.sum(h)
    return np.convolve(signal, h, mode='valid')

def estimate_time_offset(x, y, max_time_offset, upsample):
    # finds best k such that x[n+k] \approx y[k]
    y = y[max_time_offset:-max_time_offset]

    # this is a terrible way of doing this
    x = signal.resample_poly(x, upsample, 1)
    y = signal.resample_poly(y, upsample, 1)
    xcorr = np.abs(np.correlate(x, y, mode='valid'))
    return np.argmax(xcorr)/upsample - max_time_offset

class DDTimeSync:
    def __init__(self, *, max_offset, N_taps, N_filters):
        self.acquired = False
        self.time_offset = None
        self.max_offset = max_offset
        self.N_taps = N_taps
        self.history_n = []
        self.N_filters = N_filters
        self.history_offset = []
        self.n = 0
        self.fir_state = np.zeros(N_taps-1)

        # # create filterbank
        # n = np.arange(-N_taps//2, N_taps//2)+1
        # delay = np.arange(-max_offset,max_offset, 1/N_filters)
        # window = signal.windows.hamming(N_taps)
        # n = n.reshape(1,-1)
        # window = window.reshape(1,-1)
        # delay = delay.reshape(-1,1)
        # h = np.sinc(n - delay)*window
        # print(n.shape, delay.shape, h.shape)
        # h /= np.sum(h,axis=1)[:,np.newaxis]
        # self.filterbank = h

        ##################################
        # saving FIR state:
        # data:            01234567|01234567
        #                     --+--|
        # -----------------------------------
        # next block:          --+-|-
        #                       --+|--
        #                        --|+--
        #                         -|-+--
        #                          |--+--
        
    def processing_delay(self):
        return self.N_taps//2
    def __call__(self,sig,target):
        assert(len(sig) == len(target))
        self.time_offset = estimate_time_offset(
            target, sig, self.max_offset, self.N_filters
        )
        if abs(self.time_offset) == self.max_offset:
            print("[DDTimeSync] warning: max offset reached")
        self.n += len(target)
        self.history_n.append(self.n)
        self.history_offset.append(self.time_offset)
        fir_input = np.concatenate([self.fir_state, sig])
        self.fir_state = sig[-self.N_taps+1:]
        return fir_delay(fir_input, self.time_offset, self.N_taps)

# from scipy.interpolate import CubicSpline
# import matplotlib.pyplot as plt
# from time import time

# def variable_delay(sig, delay, N_up=4):
#     interpolator = CubicSpline(np.arange(N*N_up)/N_up, signal.resample_poly(sig, N_up, 1))
#     return interpolator(np.arange(len(delay))-delay)

# N = 100_000
# blocksize = 1000
#
# sig = 1/np.sqrt(2) * (np.random.normal(size=N) + 1.0j*np.random.normal(size=N))
# print("signal generated")
# n = np.arange(N)
# delay = 1.5*np.sin(2*pi*0.5 * n/N)
# target = variable_delay(sig, delay)
# target += 1/np.sqrt(400) * 1/np.sqrt(2) * (np.random.normal(size=N) + 1.0j*np.random.normal(size=N))
# print("delayed signal generated")
#
# sig_blocks = sig.reshape(-1, blocksize)
# target_blocks = target.reshape(-1, blocksize)
# time_sync = DDTimeSync(max_offset = 2, N_taps = 21, N_filters=16)
#
# t_start = time()
# out = []
# for sig_block, target_block in zip(sig_blocks, target_blocks):
#     out.append(time_sync(sig_block,target_block))
# t_stop = time()
# N_discard = time_sync.processing_delay()
# out = np.concatenate(out)[N_discard:]
# print(t_stop-t_start)
# plt.figure()
# plt.plot(abs(out-target[:-N_discard]))
#
# plt.figure()
# plt.plot(time_sync.history_n, time_sync.history_offset)
# plt.plot(np.arange(N), delay)
# plt.show()

