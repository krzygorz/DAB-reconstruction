from dab_parameters import (
    fs, N_carriers, N_frame, N_symbol,
    symbols_per_frame, zero_carrier_idx, carrier_indices
)
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_timesync():
    plt.figure()
    t = np.arange(n_frames*(symbols_per_frame-1))*N_symbol/fs
    plt.plot(t,timesync.history)
    plt.title("time drift (approximate)")
    plt.xlabel("time [s]")
    plt.ylabel("drift [samples]")

def plot_frame_diff_constellation(soft):
    plt.figure()
    k=np.arange(0,N_carriers,10)
    #c = np.broadcast_to(np.arange(symbols_per_frame-1)[:, np.newaxis], (symbols_per_frame-1,len(k)))
    c = np.tile(k, symbols_per_frame-1)
    iq_scatter = plt.scatter(soft[:,k].real, soft[:,k].imag, c=c)
    cbar = plt.colorbar(iq_scatter)
    cbar.set_label('carrier no.')
    plt.title("data differential constellation")

def plot_phase_error_histogram(soft):
    plt.figure()
    without_zero = np.delete(soft, zero_carrier_idx, axis=1).flatten()
    phase_err = np.angle(without_zero) % (pi/2) - pi/4
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
    tick_labels = [f"${x} \pi$" for x in fractions]
    plt.xticks(tick_values, tick_labels)

def ladder_plot(soft):
    fig, ax = plt.subplots()
    num_frames, _ = soft.shape
    print("num_frames", num_frames)
    y = np.angle(soft).T
    x = np.broadcast_to(carrier_indices[:, np.newaxis], y.shape)
    c = np.broadcast_to(np.arange(0,num_frames)[np.newaxis,:], y.shape)
    scatterplot = ax.scatter(x,y, s=1, c=c, cmap='Reds')
    ax.set_xlim(-N_carriers//2, N_carriers//2)
    ax.set_ylim(-pi,pi)
    ax.set_xlabel('carrier no')
    ax.set_ylabel('phase')
    ax.axhline(-pi/2, color='black', linestyle='--')
    ax.axhline(0, color='black', linestyle='--')
    ax.axhline(pi/2, color='black', linestyle='--')
    plt.title("")
    #divider   = make_axes_locatable(ax)
    #ax_stddev = divider.append_axes("top", 0.3, pad=0.1, sharex=ax)
    #stddev =np.std(np.angle(soft)%(pi/2)-pi/4,axis=0) 
    #ax_stddev.plot(carrier_indices, stddev)
    #ax_stddev.set_ylim(0, np.max(np.delete(stddev,zero_carrier_idx)))
    #ax_stddev.xaxis.set_tick_params(labelbottom=False)
    #ax_cbar = divider.append_axes("right", 0.2, pad=0.1)
    #cbar = fig.colorbar(scatterplot, cax=ax_cbar)
    cbar = fig.colorbar(scatterplot)
    cbar.set_label('frame no.')

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
    ax1.legend()
    
    eps = 1e-7
    zeros = (abs(recv) < eps) | (abs(regen) < eps)
    angles = np.angle(np.conj(recv)*regen)
    angles[zeros] = np.nan
    s = slice(0,len(angles),50)
    # ax2.plot(angles)
    ax2.scatter(np.arange(len(angles))[s], angles[s], c=abs(regen)[s], s=1)
    ax2.set_title("phase error")
    
    ax3.plot(abs(recv)-abs(regen))
    ax3r = ax3.twinx()
    ax3r.plot(np.arange(len(recv))[~zeros], abs(recv[~zeros])/abs(regen[~zeros]), color='orange')
    fig.tight_layout()

def plot_coherent_error(recv,regen):
    plt.figure()
    err = abs(recv)-abs(regen)
    plt.plot(err)
    winsize = N_symbol
    power = np.correlate(err**2, np.ones(winsize)/(winsize), mode='valid')
    plt.plot(np.sqrt(power))
