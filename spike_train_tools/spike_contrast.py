# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Tuple, Union, List
from numpy.typing import NDArray
from tqdm import tqdm

plt.rcdefaults()
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')


def spike_contrast(binary_firings : NDArray,
                   sample_rate : float,
                   dt_min : float = None,
                   dt_max : float = None,
                   N_points : int = 100,
                   stride_bin_ratio : int = 2,
                   plot : bool = False,
                   err_estimate : bool = True,
                   err_fitrange : int = 10,
                   err_fitdegree : int = 2,
                   err_plot : bool = False) -> Tuple[NDArray, NDArray, NDArray, NDArray,
                                                        Tuple[float, float], Tuple[float, float]]:
    '''
    Calculates the spike contrast from binary firing data of neurons. The function analyzes 
    the firing patterns over time and computes the spike contrast based on the temporal 
    dynamics of firing rates using a binning approach with variable time windows. 
    Implemented is in Ciba et. al (2018)

    Params:
    - binary_firings   (NDArray)        : 2D array where each row represents a neuron 
                                           and each column represents a time sample 
                                           (binary values indicating whether the neuron 
                                           fired at that time).
    - sample_rate      (float)          : Sampling rate of the firing data in Hz 
                                           (samples per second).
    - dt_min           (float, optional) : Minimum time interval for binning (in seconds). 
                                           Defaults to 0.001 seconds.
    - dt_max           (float, optional) : Maximum time interval for binning (in seconds). 
                                           Defaults to half of the total duration of the 
                                           firing data based on the sample rate.
    - N_points         (int, optional)  : Number of points to sample logarithmically 
                                           between dt_min and dt_max. Defaults to 100.
    - stride_bin_ratio (int, optional)  : Ratio used to determine the stride for binning. 
                                           Defaults to 2.
    - plot             (bool, optional) : If True, a plot of the spike contrast, contrast, 
                                           and active sparsity threshold will be generated. 
                                           Defaults to False.

    Returns:
    - spike_contrast   (NDArray)        : Calculated spike contrast values for the 
                                           given time intervals.
    - contrast         (NDArray)        : Contrast values calculated from the firing rates.
    - activeST         (NDArray)        : Active sparsity thresholds for the firing rates.
    - dts              (NDArray)        : Time intervals used for calculating the contrasts.

    Example:
    >>> spike_contrast(binary_firings, sample_rate=1000.0, dt_min=0.001, dt_max=0.5)
    '''
    
    
    N_neurons, N_samples = np.shape(binary_firings)
    
    tot_firings = np.sum(binary_firings, axis=0)

    if dt_min is None:
        dt_min = 0.001
    if dt_max is None:
        dt_max = N_samples / sample_rate / 2

    dts = np.geomspace(dt_min, dt_max, N_points)

    contrast = []
    activeST = []

    for dt in tqdm(dts, total=len(dts)):
        half_bin_size = int(dt * sample_rate / stride_bin_ratio)
        bin_edges = np.arange(0, N_samples, half_bin_size)
        bin_edges = np.append(bin_edges, [N_samples]) #add the last bin edge of the last truncated bin

        theta_k = np.zeros(len(bin_edges) - 1)
        n_k = np.zeros(len(bin_edges) - 1)

        for k in range(len(bin_edges)-stride_bin_ratio):
            theta_k[k] = np.sum(tot_firings[bin_edges[k]:bin_edges[k+stride_bin_ratio]])
            sum = np.sum(binary_firings[:,bin_edges[k]:bin_edges[k+stride_bin_ratio]], axis=1)
            n_k[k] = len(np.argwhere(sum > 0))


        contrast.append(np.sum(np.abs(np.diff(theta_k))) / (2 * np.sum(theta_k)))
        activeST.append( (np.sum(n_k*theta_k) / np.sum(theta_k) - 1) / (N_neurons-1))

    contrast = np.array(contrast)
    activeST = np.array(activeST)

    spike_contrast = contrast * activeST

    i_max = np.argmax(spike_contrast)
    sc_max = spike_contrast[i_max]
    dt_sc_max = dts[i_max]

    if plot:
        plot_spike_contrast(spike_contrast, contrast, activeST, dts)

    if err_estimate:
        sc_err, dt_err = spike_contrast_error(spike_contrast, dts, fitrange=err_fitrange, degree=err_fitdegree, plot=err_plot)
    else:
        sc_err, dt_err = None, None

    return spike_contrast, contrast, activeST, dts, (sc_max, sc_err), (dt_sc_max, dt_err)


def plot_spike_contrast(spike_contrast : NDArray,
                        contrast : NDArray,
                        activeST : NDArray,
                        dts : NDArray) -> None:
    
    _, axleft = plt.subplots(figsize=(8, 6))
    axright = axleft.twinx()
    axleft.set_title('Spike-contrast')
    plot1, = axleft.plot(dts, spike_contrast, c='k', label=r'$\mathrm{SCP}(\Delta t)$')
    plot2, = axleft.plot(dts, contrast, c='g', linestyle=':', label=r'$C(\Delta t)$')
    plot3 = axleft.scatter(dts[np.argmax(spike_contrast)], np.max(spike_contrast), label=r'$\mathrm{SC}$', color='r')
    axleft.set_ylabel(r'$C(\Delta t)$, $\mathrm{SCP}(\Delta t)$ [-]', fontsize=13)
    axleft.set_xscale('log')
    axleft.set_xlim(dts[0], dts[-1])
    axleft.grid()
    axleft.set_ylim(0, 0.3)
    axleft.set_xlabel(r'$\Delta t$ [s]', fontsize=13)
    axleft.hlines(np.max(spike_contrast), dts[0], dts[np.argmax(spike_contrast)], linestyle=':', color='r')
    axleft.vlines(dts[np.argmax(spike_contrast)], 0, np.max(spike_contrast), linestyle=':', color='r')
    leftticks = np.linspace(0, 0.3, 6)
    axleft.set_yticks(leftticks, leftticks)
    
    plot4, = axright.plot(dts, activeST, label=r'$A(\Delta t)$', c='b', linestyle='--')
    axright.set_ylabel(r'$A(\Delta t)$ [-]', rotation=270, labelpad=17, fontsize=13)
    rightticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axright.set_yticks(rightticks, rightticks)
    axright.set_xscale('log')
    axright.set_ylim(0, 1)

    plots = [plot1, plot2, plot4, plot3]
    labels = [plot.get_label() for plot in plots]
    axleft.legend(plots, labels, loc='center right')

    plt.tight_layout()
    plt.savefig('spike-contrast.pdf')
    plt.show()


def spike_contrast_error(sc : NDArray,
                         dts : NDArray,
                         fitrange : int = 10,
                         degree : int = 2,
                         plot : bool = False) -> Tuple[float, float]:
    
    i_max = np.argmax(sc)
    dts_sliced = dts[i_max-fitrange:i_max+fitrange]
    sc_sliced = sc[i_max-fitrange:i_max+fitrange]

    p = np.polyfit(dts_sliced, sc_sliced, degree)
    
    residue = sc_sliced - np.polyval(p, dts_sliced)
    SC_err = np.std(residue)
    dt_err = SC_err / (np.sqrt(4*p[0]*(p[2]-sc[i_max])+p[1]**2))

    if plot:
        _, ax = plt.subplots()
        ax.set_title('Error estimation of spike contrast')
        ax.plot(dts, sc, c='r', alpha=.52)
        ax.plot(dts_sliced, sc_sliced, c='r', label='data')
        ax.plot(dts, np.polyval(p, dts), c='k', linestyle='--', label='fit')
        ax.scatter(dts[i_max], sc[i_max], marker='x', c='k', zorder=100, label='spike contrast')

        ax.set_xlim(np.min(dts), np.max(dts))
        ax.set_xscale('log')
        ax.set_xlabel(r'$\Delta t$', fontsize=15)

        ax.set_ylim(0.9*np.min(sc), 1.1*np.max(sc))
        ax.set_ylabel(r'SCP$(\Delta t)$, SCP$_{model}(\Delta t)$', fontsize=15)

        ax.grid()

        axins = ax.inset_axes([0.18, 0.13, 0.4, 0.4], xlim=(np.min(dts_sliced), np.max(dts_sliced)), ylim=(np.min(sc_sliced), np.max(sc_sliced)), xticklabels=[], yticklabels=[])
        ax.indicate_inset_zoom(axins, edgecolor='k')
        axins.plot(np.log(dts_sliced), np.abs(residue), c='r', label="Inset Plot")
        axins.hlines(SC_err, np.min(np.log(dts_sliced)), np.max(np.log(dts_sliced)), color='g', linestyle='--')
        axins.set_ylabel(r'$\left|\mathrm{SCP}_{model} - \mathrm{SCP}\right|$', fontsize=10, backgroundcolor='w')
        axins.set_xlabel(r'$\Delta t$', fontsize=10, backgroundcolor='w')
        axins.set_yticks([0, SC_err], ['', ''])
        xmin, xmax = axins.set_xlim(np.min(np.log(dts_sliced)), np.max(np.log(dts_sliced)))
        axins.set_ylim(0, 1.05*np.max(np.abs(residue)))
        axins.text((xmin+xmax)/1.33, SC_err, r'$\sigma_{SC}$', c='g', verticalalignment='bottom', fontsize=13)
        axins.set_xticklabels([])

        plt.legend()
        plt.tight_layout()
        plt.savefig('C:\\Users\\bow-lab\\Documents\\Code\\figures\\error_estimate_spike_contrast.pdf')
        plt.show()

    return SC_err, dt_err