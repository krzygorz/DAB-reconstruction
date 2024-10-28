from dab_parameters import (
    fs, N_carriers, N_frame, N_symbol, N_guard,
    symbols_per_frame, zero_carrier_idx, carrier_indices
)
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# def plot_timesync(timesync):
#     plt.figure()
#     t = np.arange(len(timesync.history)) * N_symbol/fs
#     #t = np.arange(n_frames*(symbols_per_frame-1))*N_symbol/fs
#     plt.plot(t,timesync.history)
#     plt.title("time drift (approximate)")
#     plt.xlabel("time [s]")
#     plt.ylabel("drift [samples]")
def plot_timesync(timesync):
    plt.figure()
    plt.plot(timesync.history_n, timesync.history_offset)
    plt.title("time drift")
    plt.xlabel("time [s]")
    plt.ylabel("drift [samples]")

def plot_frame_diff_constellation(soft):
    plt.figure()
    k=np.arange(0,N_carriers,1)
    #c = np.broadcast_to(np.arange(symbols_per_frame-1)[:, np.newaxis], (symbols_per_frame-1,len(k)))
    c = np.tile(k, symbols_per_frame-1)
    iq_scatter = plt.scatter(soft[:,k].real, soft[:,k].imag, c=c, s=1)
    cbar = plt.colorbar(iq_scatter)
    cbar.set_label('carrier no.')
    plt.title("data differential constellation")

def plot_phase_error_histogram(soft):
    plt.figure()
    phase_err = np.angle(soft.flatten()) % (pi/2) - pi/4
    n_bins = 256
    _, bins, _ = plt.hist(phase_err, n_bins, density=True, label="histogram", log=True)
    mu,stddev = norm.fit(phase_err)
    plt.ylim(1e-3,2*pi)
    plt.xlim(-pi/4,pi/4)
    plt.plot(bins,norm.pdf(bins,mu,stddev), label="normal distribution fit")
    p_error = norm.cdf(-pi/2,mu,stddev)*2
    plt.title(f"phase error PDF [mu = {mu:0.4f} stddev = {stddev:0.6f}]")
    ##\n p_e = {p_error:0.2g}")
    plt.legend()
    fractions = np.array([-1/4, -1/8, 0, 1/4, 1/8])
    tick_values = fractions * np.pi
    tick_labels = [f"${x} \\pi$" for x in fractions]
    plt.xticks(tick_values, tick_labels)

def ladder_plot(soft):
    fig, ax = plt.subplots()
    num_symbols, _ = soft.shape
    y = np.angle(soft).T
    x = np.broadcast_to(carrier_indices[:, np.newaxis], y.shape)
    c = np.broadcast_to(np.arange(0,num_symbols)[np.newaxis,:], y.shape)
    scatterplot = ax.scatter(x,y, s=1, c=c, cmap='Reds')
    ax.set_xlim(-N_carriers//2, N_carriers//2)
    ax.set_ylim(-pi,pi)
    ax.set_xlabel('carrier no')
    ax.set_ylabel('phase')
    ax.axhline(-pi/2, color='black', linestyle='--')
    ax.axhline(0, color='black', linestyle='--')
    ax.axhline(pi/2, color='black', linestyle='--')
    plt.title("")
    divider   = make_axes_locatable(ax)
    ax_stddev = divider.append_axes("top", 0.3, pad=0.1, sharex=ax)
    ax_stddev.set_ylabel("amplitude")
    # stddev =np.std(np.angle(soft)%(pi/2)-pi/4,axis=0) 
    amplitude = np.mean(np.abs(soft), axis=0)
    ax_stddev.plot(carrier_indices, amplitude)
    ax_stddev.set_ylim(0, np.max(amplitude))
    ax_stddev.xaxis.set_tick_params(labelbottom=False)
    ax_cbar = divider.append_axes("right", 0.2, pad=0.1)
    cbar = fig.colorbar(scatterplot, cax=ax_cbar)
    # cbar = fig.colorbar(scatterplot)
    cbar.set_label('symbol no.')

def plot_freqsync_history(freqsync):
    plt.figure()
    n_skip = 0
    t = np.arange(n_skip,len(freqsync.history))*N_symbol/fs
    plt.plot(t,np.array(freqsync.history[n_skip:]))
    plt.title("frequency offset estimate")
    plt.ylabel("freq. offset [Hz]")
    plt.xlabel("time [s]")

def plot_regen_error(recv,regen):
    fig = plt.figure()
    ax1, ax2, ax3 = fig.subplots(3, sharex=True)
    ax1.plot(abs(recv), '-o', label="recv")
    ax1.plot(abs(regen), '-+', label="regen")
    ax1.set_title("signal amplitude")
    ax1.legend()
    
    eps = 1e-7
    zeros = (abs(recv) < eps) | (abs(regen) < eps)
    angles = np.angle(np.conj(recv)*regen)
    angles[zeros] = np.nan
    s = slice(0,len(angles),10)
    # ax2.plot(angles)
    ax2.scatter(np.arange(len(angles))[s], angles[s], c=abs(regen)[s], s=1)
    ax2.set_title("phase error")
    
    # err = abs(recv)-abs(regen)
    err = abs(recv-regen)
    ax3.plot(err, label="instantaneous")
    winsize = N_symbol
    power = np.correlate(err**2, np.ones(winsize)/(winsize), mode='valid')
    ax3.plot(np.sqrt(power), label="averaged")
    ax3.set_title("|recv(t)-regen(t)|")
    ax3.legend()
    fig.tight_layout()

def plot_regen_histogram(recv,regen):
    plt.figure()
    hist_max = 3
    d1 = np.r_[regen.real, regen.imag]
    d2 = np.r_[recv.real, recv.imag]

    H, xedges, yedges = np.histogram2d(
        # abs(recv), abs(regen),
        # range=[[0,hist_max],[0,hist_max]],
        # recv.real, regen.real,
        d1, d2,
        # range=[[-hist_max,hist_max],[-hist_max,hist_max]],
        bins=64,
    )
    # regen is axis 0
    # recv  is axis 1
    H /= np.max(H, axis=0)
    # plt.imshow(H.T, interpolation='nearest', origin='lower',
    #         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.pcolormesh(xedges, yedges, H.T)
    plt.xlim(xedges[0], xedges[-1])
    plt.ylim(yedges[0], yedges[-1])

    means = np.zeros(len(xedges)-1)
    for i in range(0,len(xedges)-1):
        e1 = xedges[i]
        e2 = xedges[i+1]
        mask = (e1 <= d1) & (d1 < e2)
        means[i] = np.mean(d2[mask])

    # plt.plot(xedges[:-1], means)
    plt.axline((0,0), slope=1, color='red', alpha=0.5)#linestyle='--')
    plt.xlabel('regen amplitude')
    plt.ylabel('recv amplitude')
    plt.title('regen linearity histogram')

def plot_regen_spectrum(recv, regen):
    plt.figure()
    plt.psd(recv,  label="recv")
    plt.psd(regen, label="regen")
    plt.legend()

def plot_regen_error_histogram(recv,regen):
    plt.figure()
    err = recv-regen
    plt.hist2d(err.real, err.imag, bins=64)
    plt.title("regen error histogram")
    plt.xlabel("real")
    plt.ylabel("imag")
