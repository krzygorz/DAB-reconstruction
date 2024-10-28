import numpy as np
def delay(signal, delay, N_taps=21, mode='same'):
    # https://pysdr.org/content/sync.html
    n = np.arange(-N_taps//2, N_taps//2)
    h = np.sinc(n - delay)
    h *= np.hamming(N_taps)
    h /= np.sum(h)
    return np.convolve(signal, h, mode=mode)

def normalize_power(x):
    return x / np.sqrt(np.mean(np.abs(x)**2))
