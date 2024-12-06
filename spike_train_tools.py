# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Literal, Tuple, Union
from numpy.typing import NDArray
from tqdm import tqdm
from scipy.special import erf
from numba import njit

plt.rcdefaults()
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')


def dim_red(rates : NDArray,
            dimension : int,
            method : Literal['PCA', 'tSNE', 'umap'] = 'PCA',
            plot : bool = True,
            ranges : NDArray = None) -> NDArray:
    '''
    Performs dimensional reduction on spike train data by mapping each unit (neuron)
    onto a separate dimension. 

    params:
    - firings_rates (NDArray)  :   array of shape (#neurons, #timestamps)
    - dimension     (int)       :   lower dimension to reduce data to 
    - method        (Literal)   :   dimension reduction algorithm to be used
                                        PCA: Principal Component Analysis
                                        t-SNE: T-Distributed sSochastic Neighbor Embedding
                                        UMAP: Uniform Manifold Approximation and Projection
    - plot          (bool)      :   determined whether results are plotted
    - ranges        (NDArray)  :   array with ranges for plot of size (dimension, 2)

    returns:    
    - components    (NDArray)  :   array of size (dimension, #timestamps)
    '''
    
    if method == 'PCA':
        model = PCA

    elif method == 'tSNE':
        model = TSNE

    elif method == 'umap':
        model = umap.UMAP
    
    # calculate reduced components
    components = model(n_components=dimension).fit_transform(rates.T)

    # plot
    if plot:
        fig = plt.figure()

        if dimension == 2:
            ax = fig.add_subplot(111)
            ax.scatter(components[:,0], components[:,1], s=5, alpha=.5, c='k')

        elif dimension == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(components[:,0], components[:,1], components[:,2], s=5, c='k', marker='o', alpha=.05)

            ax.set_zlabel('Component 3')
            if ranges is not None:
                ax.set_zlim(ranges[2][0], ranges[2][1])
                
        else:
            print('Dimension must be 2 or 3 for plotting')
        
        ax.set_title(f'{dimension}D {method} of neuronal firing data')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        if ranges is not None:
            ax.set_xlim(ranges[0][0], ranges[0][1])
            ax.set_ylim(ranges[1][0], ranges[1][1])

        plt.show()

    return components




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
                      kernel_size : float = None,
                      stride : int = None,
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
    
    if kernel_size is None:
        # pick the kernel size to be +/- 3 standard deviations
        kernel_size = 6 * sigma * sample_rate
    if stride is None:
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

  
def _make_tiling(binary_firings : NDArray,
                 sample_rate : float,
                 dt_max : float = 0.05,
                 dt_min : float = 0,
                 method : Literal['fft', 'direct'] = 'direct'):
    '''
    Creates tiling windows for binary spike trains using convolution.

    This function performs convolution on binary spike trains to create tiling windows, where each spike is expanded 
    to cover a window defined by `dt`. The result is a binary array where each spike has been broadened, making it 
    easier to assess temporal correlations between spike trains in subsequent analyses, such as the Spike Time Tiling 
    Coefficient (STTC).

    Parameters:
    - binary_firings    (NDArray)  :   2D binary array of shape (units, time points), where each row represents the spike 
                                        train of a unit (1 for spike, 0 for no spike).
    - sample_rate       (float)     :   Sampling rate of the spike train data in Hz.
    - dt                (float)     :   Tiling window width in milliseconds. This determines the size of the window applied around each spike.
                                        Default is 50 ms.

    Returns:
    - firings_conv      (NDArray)  :   2D binary array of the same shape as `binary_firings`, where each spike in the original 
                                        data has been expanded to cover the tiling window.

    Example:
    >>> tiled_spikes = _making_tiling(binary_firings, sample_rate=1000, dt=50)

    Notes:
    - The convolution uses a window of ones, with a length based on `dt` and `sample_rate`, to broaden each spike 
      to its tiling window.
    - The result is clipped between 0 and 1, maintaining binary values where each tiling window covers the time 
      around each spike.
    '''
    tiling = np.zeros_like(binary_firings)
    w_min = int(2*dt_min*sample_rate)
    w_max = int(2*dt_max*sample_rate) #window size
    
    if method == 'fft':
        assert dt_min == 0, 'fft method not available for dt_min > 0'
        window = np.ones(w_max)
        for i, train in tqdm(enumerate(binary_firings), total=np.shape(binary_firings)[0]):
            result = np.clip(np.convolve(train, window, mode='same'), 0, 1)
            tiling[i] = result

    elif method == 'direct': # much faster than fft method
        for i, train in tqdm(enumerate(binary_firings), total=np.shape(binary_firings)[0]):
            spikes = np.argwhere(train == 1)[:,0]
            for spike in spikes:
                if spike <= w_max//2:
                    tiling[i, :spike-w_min//2] = 1
                    tiling[i, spike+w_min//2+1:spike+w_max//2+1] = 1
                elif len(train) - spike <= w_max//2:
                    tiling[i, spike-w_max//2:spike-w_min//2] = 1
                    tiling[i, spike+w_min//2+1:] = 1
                else:
                    tiling[i, spike-w_max//2:spike-w_min//2] = 1
                    tiling[i, spike+w_min//2+1:spike+w_max//2+1] = 1
    return tiling


def _loop_pairs(size : int,
                sums : NDArray,
                binary_firings : NDArray,
                tiling : NDArray,
                T : NDArray) -> NDArray:
    '''
    Performs the looping over all pairs of neurons for computing the Spike Time Tiling Coefficient (STTC).
    This function is separated to include the option of compiling this loop with numba

    Params:
    - size              (int)       : size of number of neurons
    - sums              (NDArray)   : tsums of total number of spikes in spike train. 
    - binary_firings    (NDArray)   : spike trains in binary format
    - tiling            (NDArray)   : tiling
    - T                 (NDArray)   : total sum of windows (T_i)

    Returns:
    - sttc              (NDArray)  : 2D symmetric matrix of STTC values, where `sttc[i, j]` represents the STTC between units `i` and `j`.
    '''
    sttc = np.ones((size, size))
    for i in tqdm(range(size), total=size):
        for j in range(i+1, size):
            if i==0:
                sums[j] = np.sum(binary_firings[j])
            P_i = np.sum(binary_firings[i] * tiling[j])/sums[i]
            P_j = np.sum(binary_firings[j] * tiling[i])/sums[j]

            sttc[i, j] = 1/2 * ( (P_i - T[j]) / (1 - P_i*T[j]) + (P_j - T[i]) / (1 - P_j*T[i]) )
            sttc[j, i] = sttc[i, j]

    return sttc


@njit
def _loop_pairs_numba(size : int,
                sums : NDArray,
                binary_firings : NDArray,
                tiling : NDArray,
                T : NDArray) -> NDArray:
    '''
    Performs the looping over all pairs of neurons for computing the Spike Time Tiling Coefficient (STTC).
    This function is separated to include the option of compiling this loop with numba

    Params:
    - size              (int)       : size of number of neurons
    - sums              (NDArray)   : tsums of total number of spikes in spike train. 
    - binary_firings    (NDArray)   : spike trains in binary format
    - tiling            (NDArray)   : tiling
    - T                 (NDArray)   : total sum of windows (T_i)

    Returns:
    - sttc              (NDArray)  : 2D symmetric matrix of STTC values, where `sttc[i, j]` represents the STTC between units `i` and `j`.
    '''
    sttc = np.ones((size, size))
    for i in range(size):
        for j in range(i+1, size):
            if i==0:
                sums[j] = np.sum(binary_firings[j])
            P_i = np.sum(binary_firings[i] * tiling[j])/sums[i]
            P_j = np.sum(binary_firings[j] * tiling[i])/sums[j]

            sttc[i, j] = 1/2 * ( (P_i - T[j]) / (1 - P_i*T[j]) + (P_j - T[i]) / (1 - P_j*T[i]) )
            sttc[j, i] = sttc[i, j]

    return sttc


def STTC(binary_firings : NDArray,
         sample_rate : float,
         dt_max : float = 0.05,
         dt_min : float = 0,
         plot : bool = False,
         sort : bool = True,
         tiling_method : Literal['fft', 'direct'] = 'direct',
         use_numba : bool = False) -> NDArray:
    '''
    Computes the Spike Time Tiling Coefficient (STTC) for pairs of neurons in binary spike train data.
    
    The STTC is a measure of correlation between pairs of spike trains, adjusted for firing rate, and it is calculated 
    based on tiling windows defined by the parameter `dt`. This function allows the option to plot the STTC matrix,
    with an option to sort units based on summed STTC values for visualization clarity.

    Params:
    - binary_firings    (NDArray)  : 2D binary array of shape (units, time points), where each row represents the spike 
                                      train of a unit (1 for spike, 0 for no spike).
    - sample_rate       (float)     : Sampling rate of the spike train data in Hz.
    - dt                (float)     : Tiling window width in milliseconds for calculating STTC. Default is 50 ms.
    - plot              (bool)      : If True, generates a heatmap of the STTC matrix.
    - sort              (bool)      : If True and `plot` is enabled, sorts units by their summed STTC values for visualization.

    Returns:
    - sttc              (NDArray)  : 2D symmetric matrix of STTC values, where `sttc[i, j]` represents the STTC between units `i` and `j`.

    Example:
    >>> sttc_matrix = STTC(binary_firings, sample_rate=1000, dt=0.04, plot=True)

    Notes:
    - The tiling for each unit is precomputed using `_making_tiling`, and STTC is then calculated for each unique 
      unit pair (i, j) in the binary spike train data.
    - STTC values are normalized between -1 and 1, where values closer to 1 indicate higher correlation in spike timing.
    '''
    
    size = np.shape(binary_firings)[0]
    sttc = np.ones((size, size))

    N_frames = np.shape(binary_firings)[1]
    
    print('Make tilings...')
    tiling = _make_tiling(binary_firings, sample_rate, dt_max=dt_max, dt_min=dt_min, method=tiling_method)

    T = np.sum(tiling, axis=1)/N_frames

    print(f'Compute STTC with dt_max={dt_max}, dt_max={dt_min}...')
    print(f'Total number of iterations: {int((size**2-size)/2)}')

    sums = np.zeros(size)
    sums[0] = np.sum(binary_firings[0])

    if use_numba:
        sttc = _loop_pairs_numba(size, sums, binary_firings, tiling, T)
    else:
        sttc = _loop_pairs(size, sums, binary_firings, tiling, T)
    
    if plot:
        print('Plotting...')
        plot_sttc(sttc, sort=sort)

    return sttc


def plot_sttc(sttc : NDArray,
              sort : bool) -> None:
    if sort:
        order = np.argsort(np.sum(sttc, axis=0))
        sttc_sorted = sttc[order][:,order]
        plt.imshow(sttc_sorted)
    else:
        plt.imshow(sttc)
        
    plt.title('Spike Time Tiling Coefficient (STTC)')
    plt.colorbar()
    plt.xlabel('unit')
    plt.ylabel('unit')
    plt.show()


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
        N_spikes = np.random.poisson(actual_rate * T_total)
        trains_sim = (sample_rate*np.sort(T_total*np.random.random(N_spikes))).astype(int)


        trains_binary_sim[neuron][trains_sim] = True
    
    return trains_binary_sim


def spike_contrast(binary_firings : NDArray,
                   sample_rate : float,
                   dt_min : float = None,
                   dt_max : float = None,
                   N_points : int = 100,
                   stride_bin_ratio : int = 2,
                   plot : bool = False) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
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

    if plot:
        plot_spike_contrast(spike_contrast, contrast, activeST, dts)

    return spike_contrast, contrast, activeST, dts


def plot_spike_contrast(spike_contrast : NDArray,
                        contrast : NDArray,
                        activeST : NDArray,
                        dts : NDArray) -> None:
    
    _, axleft = plt.subplots(figsize=(8, 6))
    axright = axleft.twinx()

    plot1, = axleft.plot(dts, spike_contrast, c='k', label='spike contrast')
    plot2, = axleft.plot(dts, contrast, c='g', linestyle=':', label='contrast')
    plot3 = axleft.scatter(dts[np.argmax(spike_contrast)], np.max(spike_contrast), label='Maximum contrast', color='r')
    axleft.set_ylabel('Contrast [-]')
    axleft.set_xscale('log')
    axleft.set_xlim(dts[0], dts[-1])
    axleft.grid()
    axleft.set_xlabel(r'$\Delta t$ [s]')
    
    plot4, = axright.plot(dts, activeST, label='activeST', c='b', linestyle='--')
    axright.set_ylabel('ActiveST [-]', rotation=270, color='b')
    axright.set_xscale('log')
    axright.grid()

    plots = [plot1, plot2, plot3, plot4]
    labels = [plot.get_label() for plot in plots]
    axleft.legend(plots, labels)

    plt.show()