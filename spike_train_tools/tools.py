# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Tuple, Union, List
from numpy.typing import NDArray
from tqdm import tqdm
from scipy.special import erf

plt.rcdefaults()
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')



def _gaussian_kernel(size : int,
                     sigma : float) -> NDArray:
    '''
    Generates gaussian kernel

    params:
    - size    (int)      :   total number of data points of kernel
    - sigma   (float)    :   standard deviation of kernel in data points

    return
    - kernel  (NDArray) :   gaussian kernel  
    '''
    x = np.linspace(-size // 2, size // 2, int(size))
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel




def _convolution(input : NDArray,
                kernel : NDArray,
                stride : int = 1,
                padding : Literal['same', 'valid'] = 'same') -> NDArray:
    '''
    Performs a convolution with stride. 

    params:
    - input      (NDArray)  :   input that is convolved with kernel
    - kernel     (NDArray)  :   kernel to convolve input with
    - stride     (int)       :   stride
    - padding    (Literal)   :   Method of dealing with edges
                                 either same of valid

    returns:    
    - output     (NDArray)  :   convolved array
    '''
    
    input_length = len(input)
    kernel_length = len(kernel)
    output_length = (input_length - kernel_length) // stride + 1

    if padding == 'same':
        pad_size = (kernel_length - 1) // 2
        input = np.pad(input, (pad_size, pad_size), mode='constant')
    elif padding == 'valid':
        pad_size = 0
    else:
        raise ValueError("Padding must be either 'same' or 'valid'")
    
    padded_input_length = len(input)

    output_length = (padded_input_length - kernel_length) // stride + 1

    output = np.zeros(output_length)

    for i in range(0, output_length):
        start_index = i * stride
        output[i] = np.dot(input[start_index:start_index + kernel_length], kernel)

    return output




def smooth_spiketrain(trains_binary : NDArray,
                      sample_rate : float,
                      method : Literal['Direct', 'ISI'],
                      sigma : float = .05,
                      kernel_size : Union[Literal['auto'], float] = 'auto',
                      stride : Union[Literal['auto'], int] = 'auto',
                      plot : bool = False,
                      plot_nrs : NDArray = range(0, 4),
                      time_range : Tuple[float] = [0, 5]) -> NDArray:
    '''
    Smooths binary spike train data using Gaussian convolution, supporting different smoothing methods.
    
    This function performs smoothing on a binary spike train array by convolving it with a Gaussian kernel,
    where the kernel size and stride can be specified or calculated automatically. It supports two methods:
    'Direct' for a standard convolution of the spike train and 'ISI' to incorporate inter-spike intervals.

    Params:
    - binary_spike_trains   (NDArray)     : 2D array of binary spike trains, where each row represents a unit's spike 
                                             activity over time (1 for spike, 0 for no spike).
    - sample_rate           (float)        : Sampling rate of the spike train data in Hz.
    - method                (Literal['Direct', 'ISI'])
        - 'Direct': Performs a direct convolution of the binary spike train with a Gaussian kernel.
        - 'ISI': Calculates inter-spike intervals (ISI) to derive instantaneous firing rates, then smooths with convolution.
    - sigma                 (float)        : Standard deviation of the Gaussian kernel in seconds. Default is 0.05s (50 ms).
    - kernel_size           (float)        : Size of the Gaussian kernel in samples. Default is 6 * sigma * sample_rate.
    - stride                (int)          : Step size for the convolution operation, calculated if not provided.
    - plot                  (bool)         : If True, plots the binary spike trains and smoothed firing rates for specified units.
    - plot_nrs              (NDArray)     : Array of indices indicating which units to plot. Default is the first 4 units.
    - time_range            (Tuple[float]) : Time range for the x-axis of the plot, in seconds. Default is [0, 5].

    returns:
    - firings_smooth        (NDArray)     : 2D array of smoothed firing rates for each unit. Each row represents the smoothed 
                                             firing rate for one unit over time.

    Example:
    >>> smoothed_data = smooth_spiketrain(binary_spike_trains, sample_rate=1000, method='Direct', sigma=0.05)
    '''
    
    if kernel_size == 'auto':
        # pick the kernel size to be +/- 3 standard deviations
        kernel_size = 6 * sigma * sample_rate
    if stride == 'auto':
        stride = int(sigma * sample_rate / 20)

    gaussian = _gaussian_kernel(kernel_size, sigma*sample_rate)

    for i, unit in tqdm(enumerate(trains_binary), total=np.shape(trains_binary)[0]):

        if method == 'Direct':
            conv_result = _convolution(unit, gaussian, stride=stride)

        elif method == 'ISI':
            spike_train = np.argwhere(unit)[:,0]
            isi = np.diff(spike_train)
            isi_rate = sample_rate / isi

            temporal_isi_rate = np.zeros(len(unit))
            temporal_isi_rate[spike_train[:-1]] = isi_rate

            conv_result = _convolution(temporal_isi_rate, gaussian, stride=stride)

        if i == 0:
            rates = conv_result 

        else:
            rates = np.vstack((rates, conv_result))

    
    if plot:
        n_plots = len(plot_nrs)

        _, axs = plt.subplots(n_plots, 1, figsize=(10, 2.5*n_plots), sharex=True)
        plt.subplots_adjust(hspace=.07)
        plt.suptitle(f'method: {method} | '+r'$\sigma =$'+str(sigma*1e3)+' ms', y=.9)

        for i, ax in enumerate(axs):
            time = np.arange(0, len(trains_binary[i])) / sample_rate

            ax.plot(time, trains_binary[i], c='k', alpha=1, linewidth=.2)
            ax.set_yticks([0, 1])
            ax.set_ylabel('Discrete firing')
            ax.set_ylim(0)

            axright = ax.twinx()
            
            axright.plot(time[::stride], rates[i], c='r', linewidth=3)
            axright.set_ylabel('Firing rate [Hz]', rotation=270)
            axright.set_ylim(0)
            

        plt.xlabel('Time [s]')
        plt.xlim(time_range[0], time_range[1])
        plt.show()

        
    return rates


  



def _gauss(x : Union[float, NDArray],
           cen : float, sigma : float, A : float) -> Union[float, NDArray]:
    return A * np.exp(-1*(x-cen)**2/(2*sigma**2))



def _cum_gauss(x : Union[float, NDArray],
               cen : float, sigma : float) -> Union[float, NDArray]:
    return (1 + erf( (x - cen) / (sigma * np.sqrt(2)) )) / 2



def _double_gauss(x : Union[float, NDArray],
                  cen1 : float, sigma1 : float, A1 : float,
                  cen2 : float, sigma2 : float, A2 : float) -> Union[float, NDArray]:
    return _gauss(x, cen1, sigma1, A1) + _gauss(x, cen2, sigma2, A2)



def _gauss_norm(x : Union[float, NDArray],
                u : float, s : float) -> Union[float, NDArray]:
    return np.exp(-1*(x-u)**2 / (2*s**2)) / (np.sqrt(2*np.pi*s**2))
    


def simulate_spiketrains(N_neurons : int,
                         rate : Union[float, NDArray],
                         T_total : float,
                         sample_rate : int) -> NDArray:
    '''
    Generates simulated binary spike train data. It takes inter spike intervals from an 
    exponential/Poissonian distribution with a rate (lambda) which is either constant for 
    all neurons (type(rate) = float) or varies for all neurons (type(rate) = NDArray, with len(rate) == N_neurons)

    Params:
    - N_neurons     (int)               : number of neurons
    - rate          (float, NDArray)    : Poissonian distribution rate in Hz.
    - T_total       (float)             : total measurement time of the data in s
    - sample_rate   (int)               : Sampling rate of the spike train data in Hz.

    Returns:
    - trains_binary_sim  (NDArray)      : generated binary spike trains

    Example:
    >>> trains_binary_sim = simulate_spiketrains(100, 10, 60, 20000)
    '''
    
    trains_binary_sim = np.full( (N_neurons, int(T_total*sample_rate+1)), False, dtype=np.bool_)

    for i, neuron in enumerate(range(N_neurons)):
        if type(rate) != float and type(rate) != int:
            actual_rate = rate[i]
        else:
            actual_rate = rate
        N_spikes = max([1, np.random.poisson(actual_rate * T_total)]) # prevent empty spiketrains
        trains_sim = (sample_rate*np.sort(T_total*np.random.random(N_spikes))).astype(int)


        trains_binary_sim[neuron][trains_sim] = True
    
    return trains_binary_sim



def raster_plot(trains : List, 
                sample_rate : int,
                plot_avg : bool,
                time_rate_total : NDArray,
                rate_total : NDArray,
                time_range : Tuple[float] = None,
                plot_save : bool = False,
                plot_save_name : str = 'C:\\Users\\bow-lab\\Documents\\Code\\figures\\sttc.pdf') -> None:
    if time_range is None:
        time_range = (np.min(time_rate_total), np.max(time_rate_total))
    
    unit_numbers = np.arange(0, len(trains))

    _, ax1 = plt.subplots(figsize=(20, 8))

    plt.title('Neuronal activity of a primary rat culture over time', fontsize=25)

    for i, train in enumerate(trains):
        if i == 0:
            label = 'Firings'
        else:
            label = None

        ax1.scatter(train/sample_rate, 
                    np.full_like(train, unit_numbers[i]), 
                    s=.07, color='k', label=label)

    ax1.set_ylabel('Unit number [-]', fontsize=20)
    ax1.set_ylim(0, len(trains))
    ax1.set_yticks(range(0, 450, 50), range(0, 450, 50), fontsize=15)
    ax1.set_xticks(range(0, 12, 2), range(0, 12, 2), fontsize=15)

    if plot_avg:
        ax2 = ax1.twinx() 
        ax2.plot(time_rate_total, rate_total, c='r', alpha=.7, label='Population rate')
        ax2.set_ylim(0, 1.05*np.max(rate_total))
        ax2.set_ylabel('Population rate [Hz]', rotation=270, labelpad=20, fontsize=20)
        ax2.set_yticks([0, 2000, 4000, 6000, 8000], [0, 2000, 4000, 6000, 8000], fontsize=15)
    plt.legend(fontsize=20)

    ax1.set_xlabel('Time [s]', fontsize=20)
    ax1.set_xlim(time_range[0], time_range[1])
    
    if plot_save:
        plt.tight_layout()
        plt.savefig(plot_save_name)

    plt.show()



