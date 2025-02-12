# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Union
from numpy.typing import NDArray
from tqdm import tqdm
from numba import njit
import time

plt.rcdefaults()
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')


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
    with tqdm(total=size) as pbar:
        for i in range(size):
            for j in range(i+1, size):
                if i==0:
                    sums[j] = np.sum(binary_firings[j])
                
                P_i = np.sum(binary_firings[i] * tiling[j])/sums[i]
                P_j = np.sum(binary_firings[j] * tiling[i])/sums[j]
                
                #if (1 - P_i*T[j]) == 0 or (1 - P_j*T[i]) == 0:
                #    sttc[i, j] = 1
                #else:
                #    sttc[i, j] = 1/2 * ( (P_i - T[j]) / (1 - P_i*T[j]) + (P_j - T[i]) / (1 - P_j*T[i]) )
                sttc[i, j] = 1/2 * ( (P_i - T[j]) / (1 - P_i*T[j]) + (P_j - T[i]) / (1 - P_j*T[i]) )
                sttc[j, i] = sttc[i, j]
                
                pbar.update(1)

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
         use_numba : bool = False,
         plot_save : bool = False,
         plot_save_name : str = 'C:\\Users\\bow-lab\\Documents\\Code\\figures\\sttc.pdf') -> NDArray:
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
    N_frames = np.shape(binary_firings)[1]
    
    print('Make tilings...')
    tiling = _make_tiling(binary_firings, sample_rate, dt_max=dt_max, dt_min=dt_min, method=tiling_method)

    T = np.sum(tiling, axis=1)/N_frames

    print(f'Compute STTC with dt = ({dt_min}, {dt_max})')
    n_its = int((size**2-size)/2)

    sums = np.zeros(size)
    sums[0] = np.sum(binary_firings[0])

    if use_numba:
        print('<<<Progress bar not available with numba>>>')
        ect = n_its / 480
        print(f'{str(0).zfill(len(str(n_its)))}/{n_its} [{str(int(ect//60)).zfill(2)}:{str(int(ect%60)).zfill(2)}, 480 it/s]')
        start = time.time()

        try:
            sttc = _loop_pairs_numba(size, sums, binary_firings, tiling, T)
        except Exception as e:
            print(f'Numba not available due to {e}')
            sttc = _loop_pairs(size, sums, binary_firings, tiling, T)

        elapsed = time.time()-start
        print(f'{n_its}/{n_its} [{str(int(elapsed//60)).zfill(2)}:{str(int(elapsed%60)).zfill(2)}, {round(n_its/elapsed, 2)}it/s]')

    else:
        sttc = _loop_pairs(size, sums, binary_firings, tiling, T)
    
    if plot:
        print('Plotting...')
        plot_sttc(sttc, sort, plot_save, plot_save_name)


    return sttc



def plot_sttc(sttc : NDArray,
              sort : bool,
              plot_save : bool,
              plot_save_name : str) -> None:
    if sort:
        order = np.argsort(np.sum(sttc, axis=0))
        sttc_sorted = sttc[order][:,order]
        plt.imshow(sttc_sorted)
    else:
        plt.imshow(sttc)
        
    plt.title('Spike Time Tiling Coefficient (STTC)')
    plt.colorbar()#label=r'$\mathrm{STTC}(S_i,\,S_j,\,\Delta t)$')
    plt.xlabel('Units')
    plt.ylabel('Units')
    if plot_save:
        plt.tight_layout()
        plt.savefig(plot_save_name)
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
