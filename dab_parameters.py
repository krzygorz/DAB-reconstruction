import numpy as np
from numpy.fft import fft, ifft

table_23 = np.array([
    [-768, -737, 0, 1],
    [-736, -705, 1, 2],
    [-704, -673, 2, 0],
    [-672, -641, 3, 1],
    [-640, -609, 0, 3],
    [-608, -577, 1, 2],
    [-576, -545, 2, 2],
    [-544, -513, 3, 3],
    [-512, -481, 0, 2],
    [-480, -449, 1, 1],
    [-448, -417, 2, 2],
    [-416, -385, 3, 3],
    [-384, -353, 0, 1],
    [-352, -321, 1, 2],
    [-320, -289, 2, 3],
    [-288, -257, 3, 3],
    [-256, -225, 0, 2],
    [-224, -193, 1, 2],
    [-192, -161, 2, 2],
    [-160, -129, 3, 1],
    [-128,  -97, 0, 1],
    [-96,   -65, 1, 3],
    [-64,   -33, 2, 1],
    [-32,    -1, 3, 2],
    [  1,    32, 0, 3],
    [ 33,    64, 3, 1],
    [ 65,    96, 2, 1],
    [ 97,	128, 1, 1],
    [ 129,  160, 0, 2],
    [ 161,  192, 3, 2],
    [ 193,  224, 2, 1],
    [ 225,  256, 1, 0],
    [ 257,  288, 0, 2],
    [ 289,  320, 3, 2],
    [ 321,  352, 2, 3],
    [ 353,  384, 1, 3],
    [ 385,  416, 0, 0],
    [ 417,  448, 3, 2],
    [ 449,  480, 2, 1],
    [ 481,  512, 1, 3],
    [ 513,  544, 0, 3],
    [ 545,  576, 3, 3],
    [ 577,  608, 2, 3],
    [ 609,  640, 1, 0],
    [ 641,  672, 0, 3],
    [ 673,  704, 3, 0],
    [ 705,  736, 2, 1],
    [ 737,  768, 1, 1],
])

h_table = np.array([
  [0, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 1, 1,
   0, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 1, 1],
  
  [0, 3, 2, 3, 0, 1, 3, 0, 2, 1, 2, 3, 2, 3, 3, 0,
   0, 3, 2, 3, 0, 1, 3, 0, 2, 1, 2, 3, 2, 3, 3, 0],
  
  [0, 0, 0, 2, 0, 2, 1, 3, 2, 2, 0, 2, 2, 0, 1, 3,
   0, 0, 0, 2, 0, 2, 1, 3, 2, 2, 0, 2, 2, 0, 1, 3],
   
  [0, 1, 2, 1, 0, 3, 3, 2, 2, 3, 2, 1, 2, 1, 3, 2,
   0, 1, 2, 1, 0, 3, 3, 2, 2, 3, 2, 1, 2, 1, 3, 2]
])

fs = 2048000
N_carriers = 1536 
N_frame = 196608
N_null = 2656
N_symbol = 2552
N_guard = 504
N_fft = N_symbol-N_guard
symbols_per_frame = 76
carrier_spacing = fs/N_fft #1kHz

ref_sync_carriers = np.zeros(N_fft, dtype="complex128")
for k in range(-N_carriers//2,N_carriers//2+1):
   idx = k if k >=0 else N_fft+k
   if k == 0:
       ref_sync_carriers[idx] = 0
       continue
   k1,_, i, n = table_23[(table_23[:,0] <= k) & (k <= table_23[:,1])][0]
   phase = np.pi/2 * (h_table[i, k-k1] + n)
   ref_sync_carriers[idx] = np.exp(1.0j * phase)

def add_guard(symbol):
    return np.concatenate([symbol[-N_guard:], symbol])
def mod_ofdm_symbol(carriers):
    return add_guard(ifft(carriers))

ref_sync_symbol = mod_ofdm_symbol(ref_sync_carriers)

zero_carrier_idx = N_carriers//2
def remove_guard(symbol):
    offset = 0
    return symbol[N_guard-offset:N_symbol-offset]
# import matplotlib.pyplot as plt
def get_data_carriers(symbol):
    all_carriers = fft(remove_guard(symbol))
    # plt.figure()
    # plt.plot(np.angle(all_carriers), 'o')
    return np.r_[all_carriers[-N_carriers//2:], all_carriers[1:N_carriers//2+1]]
    # ind = np.r_[N_fft-N_carriers//2+np.arange(N_carriers//2), 1+np.arange(N_carriers//2)]
    # plt.plot(ind, np.angle(cut), 'x')
    # plt.show()
    # return cut
def insert_null_carriers(data_carriers):
    negative = data_carriers[:N_carriers//2]
    positive = data_carriers[N_carriers//2:]
    carriers = np.zeros(N_fft, dtype=np.complex64)
    carriers[1:N_carriers//2+1] = positive
    carriers[-N_carriers//2:] = negative
    return carriers

# from -N_carriers to N_carriers, inclusive, but without 0 in the middle
carrier_indices = np.r_[np.arange(-N_carriers//2,0), np.arange(1,N_carriers//2+1)]
