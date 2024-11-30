# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import numpy as np
import matplotlib.pyplot as plt
import spike_train_tools as stt
import pickle

from typing import Tuple, Literal, List
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

plt.rcdefaults()
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')

class MEAData():
    '''
    Data class for managing MEA data

    params:
    - path      (str)    :   file path to .npz data file
    - overview  (bool)   :   bool determining if overview of npz file is printing

    Example:
    >>> mea_data = MEAData('.\\sorted.npy', overview=True)
    '''
    def __init__(self, path : str,
                 file_type : Literal['raw', 'processed'],
                 overview : bool = True,
                 pressure : float = None) -> None:
    
        self.file_type = file_type

        if self.file_type == 'raw':
            assert path.split('.')[-1] == 'npz', 'raw input file should be an npz file'

            self.file_ID = path.split('\\')[-2]
            self.data = np.load(path, allow_pickle=True)
            self.units = self.data['units']
            self.locations = self.data['locations']
            self.sample_rate = int(self.data['fs'])
            self.pressure = pressure

            self.trains = self.get_trains()
            self.locs = self.get_locs()

            self.N_units = len(self.units)

            self.N_samples = self.get_N_samples()
            self.T_total = self.N_samples / self.sample_rate

            self.trains_binary = self.convert_trains_to_binary()

            self.initialize_attributes()


        elif self.file_type == 'processed':
            assert path.split('.')[-1] == 'pkl', 'processed input file should be an pkl file'

            self.units = None
            self.locations = None

            with open(path, 'rb') as file:
                self.data = pickle.load(file)

            self.parse_processed_data()

            self.trains_sorted = None
            
        else:
            print('file type must be either \'raw\' or \'processed\'')

        self.unit_numbers = np.arange(0, self.N_units)

        if overview:
            self.print_data_overview()


    def initialize_attributes(self) -> None:
        self.firings_total = None
        self.rate_total = None
        self.time_rate_total = None
        self.rates = None
        self.time_rates = None
        self.rates_avg = None
        self.trains_sorted = None
        self.sttc = None
        self.sttc_dt = None
        self.sim = None
        self.sttc_sim = None
        self.correlation = None
        self.SC = None
        self.contrast = None
        self.activeST = None
        self.SC_dts = None
        self.SC_max = None
        self.SC_dt_max = None
        self.sttc_clustered = None
        self.cluster_labels = None
        self.cluster_centers = None


    def parse_processed_data(self) -> None:
        if self.data['mode'] == 'minimal':
            print('processed data saved in minimal mode cannot be reloaded in MEAData object')
            return 
        
        self.pressure = self.data['pressure']
        self.file_ID = self.data['file_ID']
        self.N_units = self.data['N_units']
        self.N_samples = self.data['N_samples']
        self.T_total = self.data['T_total']
        self.sample_rate = self.data['sample_rate']
        self.locs = self.data['locs']
        self.trains = self.data['trains']
        self.sttc = self.data['sttc']
        self.sttc_dt = self.data['sttc_dt']
        self.sim = self.data['sim']
        self.sttc_sim = self.data['sttc_sim']
        self.correlation = self.data['correlation']
        self.SC_max = self.data['SC_max']
        self.SC_dt_max = self.data['SC_dt_max']
        self.sttc_clustered = self.data['sttc_clustered']
        self.cluster_labels = self.data['cluster_labels']
        self.cluster_centers = self.data['cluster_centers']

        if self.data['mode'] == 'condensed':
            self.trains_binary = self.convert_trains_to_binary()
            self.firings_total = None
            self.rate_total = None
            self.time_rate_total = None
            self.rates = None
            self.time_rates = None
            self.rates_avg = None
            self.SC = None
            self.contrast = None
            self.activeST = None
            self.SC_dts = None

        elif self.data['mode'] == 'complete':
            self.trains_binary = self.data['trains_binary']
            self.firings_total = self.data['firings_total']
            self.rate_total = self.data['rate_total']
            self.time_rate_total = self.data['time_rate_total']
            self.rates = self.data['rates']
            self.time_rates = self.data['time_rates']
            self.rates_avg = self.data['rates_avg']
            self.SC = self.data['SC']
            self.contrast = self.data['contrast']
            self.activeST = self.data['activeST']
            self.SC_dts = self.data['SC_dts']       


    def get_trains(self):
        if self.file_type == 'raw':

            trains = []

            for unit in self.units:
                trains.append(unit['spike_train'])

            return trains
        
        else:
            return self.trains


    def get_locs(self) -> Tuple[NDArray]:
        if self.file_type == 'raw':
            xlocs = []
            ylocs = []

            for unit in self.units:
                xlocs.append(unit['x_max'])
                ylocs.append(unit['y_max'])

            return np.array([xlocs, ylocs]).T
        
        else:
            return self.locs
        

    def get_N_samples(self) -> int:
        '''
        Calculates and returns the total data size across all units.

        Returns:
        - total_size (int) : total size
        '''
        total_size = 0

        for train in self.trains:
            try:
                # see if last index of spiketrain is bigger than current total size
                if train[-1] > total_size:
                    total_size = train[-1]
            except:
                continue

        return int(total_size)


    def print_data_overview(self,
                            spacing : int = 18,
                            value_length : int = 140) -> None:
        '''
        Prints an overview of the MEA data. 
        '''

        keys = list(self.data.keys())

        # PRINT HEADER
        print('KEY'.ljust(spacing),
              'SHAPE'.ljust(spacing),
              'TYPE'.ljust(spacing),
              'VALUE')

        for key in keys:
            content = self.data[key]
            try:
                content_shape = str(np.shape(content))
            except ValueError:
                content_shape = 'inhomogeneous'

            content_type = type(content).__name__
            print(key.ljust(spacing),
                  content_shape.ljust(spacing),
                  content_type.ljust(spacing),
                  repr(content).replace('\n', '').ljust(value_length)[:value_length])

        if self.file_type == 'raw':
            print()
            print('Units consists of an array of dictionary with more data', end='\n\n')

            units = self.data['units'] 
            units_keys = units[0].keys()

            for key in units_keys:
                content = units[0][key]
                try:
                    content_shape = str(np.shape(content))
                except ValueError:
                    content_shape = 'inhomogeneous'

                content_type = type(content).__name__    
                print(key.ljust(spacing),
                      content_shape.ljust(spacing),
                      content_type.ljust(spacing),
                      repr(content).replace('\n', '').ljust(value_length)[:value_length])
        print()       
        print(f'Total measurement time: {int(self.T_total//60)} min {round(self.T_total%60, 3)} s', end='\n\n')


    def convert_trains_to_binary(self) -> NDArray:
        '''
        Converts spike times into a binary array representation of size (#neurons, total_size)
        where for each time stamp, for each neuron a firing event is denoted by a True value

        Returns:
        - binary_firings (NDArray) : array of size (#neurons, self.total_size)
        '''
        # Create array with False values
        binary_firings = np.full( (self.N_units, self.N_samples+1), False, dtype=np.bool_)

        for i, train in enumerate(self.trains):
            # change indices from spike train to True
            binary_firings[i][train] = True

        return binary_firings
    

    def get_firings_total(self) -> NDArray:
        '''
        Calculates and returns the accumulated firing rate across all units.

        Returns:
        - total_firing_rate (NDArray) : array of length (self.total_size) with the 
                                         total firing rate at each timestamp
        '''
        if self.trains_binary is None:
            self.trains_binary = self.convert_trains_to_binary()

        # sum binary firings along all neurons to get total firing rate
        self.firings_total = np.sum(self.trains_binary, axis=0)

        return self.firings_total


    def get_rate_total(self, window_size : float = 0.03,
                       stride : int = None) -> Tuple[NDArray]:
        '''
        Computes and returns the time-averaged total firing rate.
        
        Params:
        - window_size   (float) : Width of the averaging window in seconds.
        - step          (int)   : Step size for moving the window, defaults to 5% of window_size.

        Returns:
        - total_firing_rate_avg (NDArray) : averaged total firing rate
        - time_avg              (NDArray) : corresponding timestamps
        '''
        # calculate non-average total firing rate
        if self.firings_total is None:
            self.get_firings_total()

        window = self.sample_rate * window_size
        if stride == None:
            stride = window / 20
        
        # get the total time array
        time = np.arange(0, len(self.firings_total)/self.sample_rate, 1/self.sample_rate)

        total_firing_rate_avg = []
        time_avg = []

        # perform discrete convolution with a square window
        for i in range(0, len(self.firings_total), int(stride)):

            total_firing_rate_avg.append(np.sum(self.firings_total[i:i+int(window)])/window_size)
    
            time_avg.append(time[i]+window_size/2)

        self.rate_total = np.array(total_firing_rate_avg)
        self.time_rate_total = np.array(time_avg)

        return self.rate_total, self.time_rate_total
    

    def sort_trains(self) -> List:
        '''
        Sorts units by activity level (number of spikes over full time window) and returns the sorted list.

        Returns
        - units_sorted (List) : sorted units
        '''
        activity = []
        for train in self.trains:
            activity.append(len(train))

        order = np.argsort(-1*np.array(activity))

        self.trains_sorted = []

        for index in order:
            self.trains_sorted.append(self.trains[index])

        return self.trains_sorted
    

    def raster_plot(self,
                    sort : bool = True,
                    plot_avg : bool = True,
                    time_range : Tuple[float] = None) -> None:
        '''
        Plots neuron activity over time, with options for sorted data and average firing rate.
        
        Params:
        - sort          (bool)          : If True, sorts units by activity.
        - plot_avg      (bool)          : If True, includes a average firing rate plot.
        - time_range    (Tuple[float])  : Range of time (in minutes) for the x-axis.
        '''
        
        if sort:
            if self.trains_sorted is None:
                self.sort_trains()
            plot_trains = self.trains_sorted

        else:
            plot_trains = self.trains
        
        if plot_avg and self.rate_total is None:
            self.get_rate_total()

        if time_range is None:
            time_range = (np.min(self.time_rate_total), np.max(self.time_rate_total))

        _, ax1 = plt.subplots(figsize=(20, 8))

        plt.title('Neuron Activity over time')

        for i, train in enumerate(plot_trains):
            if i == 0:
                label = 'Firings'
            else:
                label = None

            ax1.scatter(train/self.sample_rate, 
                        np.full_like(train, self.unit_numbers[i]), 
                        s=.07, color='k', label=label)

        ax1.set_ylabel('Unit number [-]')
        ax1.set_ylim(0, len(plot_trains))

        if plot_avg:
            ax2 = ax1.twinx() 
            ax2.plot(self.time_rate_total, self.rate_total, c='r', alpha=.7, label='Firing rate')
            ax2.set_ylim(0, 1.05*np.max(self.rate_total))
            ax2.set_ylabel('Firing rate [Hz]', rotation=270)
        plt.legend()

        plt.xlabel('Time [min]')
        plt.xlim(time_range[0], time_range[1])
        plt.show()


    def get_rates(self, method : Literal['Direct', 'ISI'],
                  sigma : float = .05,
                  kernel_size : float = None,
                  stride : int = None,
                  plot : bool = False,
                  plot_nrs : NDArray = range(0, 4),
                  time_range : Tuple[float] = [0, 5]) -> NDArray:
        '''
        Smooths firing rates and calculates the average firing rate using a specified method.
        Direct: applies Gaussian smoothing directly to the binary spike train data
        ISI: scales all spikes in the spike train by the Inter Spike Interval of the spike
        and the next spike, and then applies Gaussian Smoothing. This makes the final smoothed 
        spike train less susceptible to incidental spikes resulting from errors in KiloSort
        
        Parameters:
        - method        (str)   : Method for smoothing, either 'Direct' or 'ISI'.
        - sigma         (float) : Standard deviation of the Gaussian kernel.
        - kernel_size   (float) : Size of the smoothing kernel.
        - stride        (int)   : Step size for the convolution.
        - plot          (bool)  : If True, plots the smoothed firing rates.

        Returns:
        - firing_rate_avg   (NDArray)  : array of size (#neurons, total_size // stride)
        '''
        
        if self.trains_binary is None:
            self.trains_binary = self.convert_trains_to_binary()
        
        self.rates = stt.smooth_spiketrain(self.trains_binary, self.sample_rate,
                                           method, sigma, kernel_size, stride,
                                           plot, plot_nrs, time_range)
        
        self.time_rates = np.linspace(0, self.T_total, len(self.rates))
        
        return self.rates, self.time_rates


    def get_rates_avg(self) -> NDArray:
        if self.trains_binary is None:
            self.trains_binary = self.convert_trains_to_binary()
        
        self.rates_avg = np.sum(self.trains_binary, axis=1)/self.T_total

        return self.rates_avg


    def dim_red(self, dimension : int,
                method : Literal['PCA', 'tSNE', 'umap'] = 'PCA',
                plot : bool = True,
                ranges : NDArray = None) -> NDArray:
        '''
        Reduces the dimensionality of the firing rate data and returns components.
        
        Parameters:
        - dimension     (int)   : Target dimensionality (2 or 3).
        - method        (str)   : Dimensionality reduction technique, 'PCA', 'tSNE', or 'umap'.
        - plot          (bool)  : If True, plots the reduced data.

        Returns:
        - components (NDArray) : principal components
        '''
        if self.rates is None:
            print('First calculate smoothened firing rates with get_avg_firing_rate()')
            return
        
        components = stt.dim_red(self.rates, dimension, method, plot, ranges)

        return components
    

    def compute_sttc(self, dt_max : float = 0.05,
                     dt_min : float = 0,
                     overwrite : bool = False,
                     plot : bool = False,
                     sort : bool = True,
                     tiling_method : Literal['fft', 'direct'] = 'direct') -> NDArray:
        '''
        Computes and returns the Spike Time Tiling Coefficient (STTC) for neuron pairs.
        
        Parameters:
        - dt    (float) : Tiling window width in milliseconds.
        - plot  (bool)  : If True, generates a heatmap of STTC values.
        - sort  (bool)  : If True, sorts units by STTC sum for plotting.

        Returns:
        - sttc_result (NDArray) : array of size (#neurons, #neurons) yielding sttc
        '''
        
        if self.trains_binary is None:
            self.trains_binary = self.convert_trains_to_binary()
        if self.sttc is not None and overwrite:
            print('Warning: overwriting STTC results stored in MEAData object!')


        sttc = stt.STTC(self.trains_binary, self.sample_rate,
                             dt_max=dt_max, dt_min=dt_min, plot=plot,
                             sort=sort, tiling_method=tiling_method)
        

        sttc_dt = (dt_min, dt_max)
        
        if not overwrite:
            if not type(self.sttc) == list:
                self.sttc = [self.sttc]
                self.sttc_dt = [self.sttc_dt]

            self.sttc.append(sttc)
            self.sttc_dt.append(sttc_dt)

        else:
            self.sttc = sttc
            self.sttc_dt = sttc_dt
            
        return sttc


    def get_clusters(self, n_clusters : int,
                     plot : bool = True) -> Tuple[NDArray, NDArray]:
        
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.locs)

        self.cluster_labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_

        if plot:
            self.plot_clusters(self.cluster_labels, self.cluster_centers)

        return self.cluster_labels, self.cluster_centers


    def plot_clusters(self, labels : NDArray = None,
                      centers : NDArray = None) -> None:

        if labels is None:
            if self.cluster_labels is not None:
                labels = self.cluster_labels
            else:
                print('No cluster labels defined, run get_cluster() first')
        
        if centers is None:
            if self.cluster_centers is not None:
                centers = self.cluster_centers
            else:
                print('No cluster centers defined, run get_cluster() first')

        cmap = plt.get_cmap("tab20", len(centers))

        _, axs = plt.subplots(1, 2, figsize=(13, 20))

        for cluster in range(len(centers)):
            points = self.locs[labels == cluster]
            axs[0].scatter(points[:, 0], points[:, 1], color=cmap(cluster), marker='s')
            axs[0].text(centers[cluster, 0], centers[cluster, 1], str(cluster))

        axs[0].set_aspect('equal')
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        axs[0].set_title("K-Means Clustering")
        axs[0].grid(True)
        
        # Make a pixelated image
        xrange = (np.min(self.locs[:,0]), np.max(self.locs[:,0]))
        xsize = (xrange[1] - xrange[0])
        yrange = (np.min(self.locs[:,1]), np.max(self.locs[:,1]))
        ysize = (yrange[1] - yrange[0])

        pixels = np.zeros( (int(2*xsize)+1, int(2*ysize)+1) )

        for loc in self.locs:
            pixels[int(2*(loc[0]-xrange[0])), int(2*(loc[1]-yrange[0]))] = np.inf

        axs[1].imshow(np.rot90(pixels, k=1))
        axs[1].set_title('Pixel image')

        plt.show()


    def get_STTC_clustered(self, plot : bool = True) -> NDArray:
        if self.cluster_labels is None:
            self.get_clusters()

        if self.sttc is None:
            self.compute_sttc()

        order = np.argsort(self.cluster_labels)
        sttc_result_ordered = self.sttc[order][:,order]
        labels_ordered = self.cluster_labels[order]
        np.fill_diagonal(sttc_result_ordered, np.nan)

        self.sttc_clustered = np.zeros( (len(self.cluster_centers), len(self.cluster_centers)) )

        for i in range(len(self.cluster_centers)):
            for j in range(len(self.cluster_centers)):
                if i >= j:
                    sttc_cluster = np.nanmean(sttc_result_ordered[labels_ordered == i][:,labels_ordered == j])
            
                    self.sttc_clustered[i, j] = sttc_cluster
                    self.sttc_clustered[j, i] = sttc_cluster

        if plot:
            self.plot_STTC_clustered()

        return self.sttc_clustered

        
    def plot_STTC_clustered(self):
        if self.sttc_clustered is None:
            self.get_STTC_clustered()
        elif int(np.shape(self.sttc_clustered)[0]) != int(len(self.cluster_centers)):
            self.get_STTC_clustered()

        order = np.argsort(self.cluster_labels)
        sttc_result_ordered = self.sttc[order][:,order]

        tick_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        lines = np.cumsum(counts)-.5
        ticks = lines - counts/2 

        fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharex=True, sharey = True)

        axs[0].imshow(sttc_result_ordered)
        axs[0].set_title('STTC')
        axs[0].set_aspect('equal')
        axs[0].hlines(lines, -.5, len(self.units)-.5, color='r', linewidth=1)
        axs[0].vlines(lines, -.5, len(self.units)-.5, color='r', linewidth=1)
        axs[0].set_xticks(ticks, tick_labels, fontsize=7)
        axs[0].set_yticks(ticks, tick_labels, fontsize=7)


        tick_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        lines = np.cumsum(counts)

        edges = np.hstack([[0], lines])

        diffs = np.diff(edges)
        ticks = lines - diffs/2
        
        mesh = axs[1].pcolormesh(edges, edges, self.sttc_clustered,
                                 shading='auto', cmap='viridis', edgecolors='r', linewidths=.7)

        cbar = fig.colorbar(mesh, ax=axs, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('STTC')

        axs[1].set_xlabel("Cluster")
        axs[1].set_ylabel("Cluster")
        axs[1].set_title("STTC averaged over clusters")
        axs[1].set_aspect('equal')
        axs[1].set_xticks(ticks, tick_labels, fontsize=7)
        axs[1].set_yticks(ticks, tick_labels, fontsize=7)

        plt.show()


    def get_sim(self, dt_max : float = 0.05,
                dt_min : float = 0):
        
        if self.rates_avg is None:
            self.get_rates_avg()

        self.sim = stt.simulate_spiketrains(self.N_units, self.rates_avg, self.T_total, self.sample_rate)
        
        self.sttc_sim = stt.STTC(self.sim, sample_rate=self.sample_rate, dt_max=dt_max, dt_min=dt_min, plot=False)

        return self.sim, self.sttc_sim


    def compare_STTC_to_random(self, dt_max : float = 0.05,
                               dt_min : float = 0,
                               N_bins : int = 120,
                               plot : bool = True,
                               method : Literal['pdf_difference',
                                                'Cramer-von Mises_data',
                                                'Wasserstein_data',
                                                'Cramer-von Mises_fit',
                                                'Wasserstein_fit',
                                                'all'] = 'Wasserstein_fit',
                               overwrite : bool = False,
                               plot_sim_hist = True) -> float:
    
        if self.sttc is None:
            print('Compute STTC on data with dt = 0.05...')
            sttcs = self.compute_sttc(dt_max, dt_min)
        elif type(self.sttc) == list:
            if overwrite:
                sttcs = self.sttc[-1]
            else:
                sttcs = self.sttc
        else:
            sttcs = [self.sttc]

        self.correlation = []
            
        for sttc in sttcs:
            result = sttc[np.triu_indices(np.shape(sttcs)[0], k=1)]

            print('Performing simulation...')
            self.get_sim(dt_max, dt_min)
            
            sim = self.sttc_sim[np.triu_indices(np.shape(self.sttc_sim)[0], k=1)]

            total_counts = len(result)

            if total_counts != len(sim):
                print('Length of data and sim don\'t match up.')

            bin_min = min([np.min(result), np.min(sim)])
            bin_max = max([np.max(result), np.max(sim)])

            bin_size = (bin_max - bin_min) / N_bins
            bin_edges = np.arange(bin_min, bin_max + bin_size, bin_size)
            bin_centers = bin_edges[1:] - bin_size / 2

            counts, _ = np.histogram(result, bins=bin_edges)
            counts_sim, _ = np.histogram(sim, bins=bin_edges)

            counts_norm = counts / total_counts / bin_size
            counts_sim_norm = counts_sim / total_counts / bin_size

            # fit gauss to simulation
            popt, pcov = curve_fit(stt._gauss_norm, bin_centers, counts_sim_norm, p0=[np.mean(sim), np.std(sim)])
            err = np.sqrt(np.diag(pcov))

            counts_cum = np.cumsum(counts_norm) * bin_size
            counts_sim_cum = np.cumsum(counts_sim_norm) * bin_size

            if plot:
                self.plot_compare_sttc_to_random(bin_centers, popt, err,
                                                counts_norm, counts_cum,
                                                counts_sim_norm, counts_sim_cum,
                                                plot_sim_hist)


            if method == 'pdf_difference':
                slice = (counts_sim_norm > counts_norm)
                self.correlation.append(np.sum(np.abs(counts_sim_norm[slice] - counts_norm[slice])))
            elif method == 'Cramer-von Mises_data':
                self.correlation.append(np.sqrt(np.sum((np.abs(counts_cum - counts_sim_cum))**2)))
            elif method == 'Wasserstein_data':
                self.correlation.append(np.sum(np.abs(counts_cum - counts_sim_cum)))
            elif method == 'Cramer-von Mises_fit':
                self.correlation.append(np.sqrt(np.sum((np.abs(counts_cum - stt._cum_gauss(bin_centers, popt[0], popt[1])))**2)))
            elif method == 'Wasserstein_fit':
                self.correlation.append(np.sum(np.abs(counts_cum - stt._cum_gauss(bin_centers, popt[0], popt[1]))))
            elif method == 'all':
                slice = (counts_sim_norm > counts_norm)
                corr1 = np.sum(np.abs(counts_sim_norm[slice] - counts_norm[slice]))
                corr2 = np.sqrt(np.sum((np.abs(counts_cum - counts_sim_cum))**2))
                corr3 = np.sum(np.abs(counts_cum - counts_sim_cum))
                corr4 = np.sqrt(np.sum((np.abs(counts_cum - stt._cum_gauss(bin_centers, popt[0], popt[1])))**2))
                corr5 = np.sum(np.abs(counts_cum - stt._cum_gauss(bin_centers, popt[0], popt[1])))
                self.correlation.append([corr1, corr2, corr3, corr4, corr5])
        
        return self.correlation


    def plot_compare_sttc_to_random(self, bin_centers : NDArray,
                                    popt : NDArray,
                                    err : NDArray,
                                    counts_norm : NDArray,
                                    counts_cum : NDArray,
                                    counts_sim_norm : NDArray,
                                    counts_sim_cum : NDArray,
                                    plot_sim_hist : bool) -> None:
        
        bin_size = np.mean(np.diff(bin_centers))
        bin_min, bin_max = bin_centers[0]-bin_size/2, bin_centers[-1]+bin_size/2
        bin_continuous = np.linspace(bin_min, bin_max, 1000)
        gauss = stt._gauss_norm(bin_centers, *popt)
        gauss_cont = stt._gauss_norm(bin_continuous, *popt)

        cum_gauss_cont = stt._cum_gauss(bin_continuous, *popt)
        cum_gauss = stt._cum_gauss(bin_centers, *popt)

        _, axs = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle(f'Xenon pressure: {self.pressure} psi')

        if plot_sim_hist:
            axs[0].bar(bin_centers, counts_sim_norm, width=bin_size, color='white', edgecolor='k', alpha=.5, label='Poisson')
            axs[1].bar(bin_centers, counts_sim_cum, width=bin_size, color='white', edgecolor='k', alpha=.5, label='random sim')

        axs[0].set_title('pdf')
        axs[0].bar(bin_centers, counts_norm, width=bin_size, color='r', edgecolor='k', alpha=.5, label='data')
        axs[0].plot(bin_continuous, gauss_cont, c='k', linestyle='--', label='fit')
        axs[0].fill_between(bin_centers, gauss, counts_norm, where=(counts_norm > gauss), 
                color='b', alpha=.3, label='correlation')
        axs[0].set_ylabel('Probability density')
        axs[0].text(.96*bin_max, .7 / np.sqrt(2*np.pi*popt[1]**2),
                    f'µ = {popt[0]:.4f} ± {err[0]:.4f}\nσ = {popt[1]:.4f} ± {err[1]:.4f}', 
                    horizontalalignment='right', verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        axs[0].legend()

        axs[1].set_title('cdf')
        axs[1].fill_between(bin_centers, counts_cum, cum_gauss, where=(cum_gauss > counts_cum), 
                color='b', alpha=0.3)
        axs[1].fill_between(bin_centers, counts_cum, cum_gauss, where=(cum_gauss <= counts_cum), 
                color='b', alpha=0.3)
        axs[1].bar(bin_centers, counts_cum, width=bin_size, color='r', edgecolor='k', alpha=.5, label='data')
        axs[1].plot(bin_continuous, cum_gauss_cont, c='k', linestyle='--', label='fit')
        axs[1].set_ylabel('Cumulative probability density')
        

        for ax in axs:
            ax.set_xlabel('STTC [-]')
            ax.set_xlim(bin_min, bin_max)
        plt.show()


    def spike_contrast(self, dt_min : float = None,
                       dt_max : float = None,
                       N_points : int = 100,
                       stride_bin_ratio : int = 2,
                       plot : bool = False) -> Tuple[float, float]:
        
        if self.SC_max is not None:
            print('Warning: overwriting current spike contrast results')
        
        self.SC, self.contrast, self.activeST, self.SC_dts = stt.spike_contrast(self.trains_binary,
                                                                                self.sample_rate,
                                                                                dt_min, dt_max, N_points,
                                                                                stride_bin_ratio, plot)
        
        self.SC_max = np.max(self.SC)
        self.SC_dt_max = self.SC_dts[np.argmax(self.SC)]

        return self.SC_max, self.SC_dt_max


    def run_all(self) -> None:
        if self.firings_total is None:
            self.get_firings_total()
        if self.rate_total is None:
            self.get_rate_total()
        if self.rates is None:
            self.get_rates()
        if self.rates_avg is None:
            self.get_rates_avg()
        if self.sttc is None:
            self.compute_sttc()
        if self.sttc_clustered is None:
            self.get_STTC_clustered()
        if self.sim is None:
            self.get_sim()
        if self.sttc_sim is None or self.correlation is None:
            self.compare_STTC_to_random(plot=False, method='all')


    def convert_results_to_dict(self,
                                mode : Literal['condensed', 'complete', 'minimal']) -> dict:
        if mode == 'condensed':
            results_dict = {
                'mode'            :   mode,
                'pressure'        :   self.pressure,
                'file_ID'         :   self.file_ID,
                'N_units'         :   self.N_units,
                'N_samples'       :   self.N_samples,
                'T_total'         :   self.T_total,
                'sample_rate'     :   self.sample_rate,
                'locs'            :   self.locs,
                'trains'          :   self.trains,
                'sttc'            :   self.sttc,
                'sttc_dt'         :   self.sttc_dt,
                'sim'             :   self.sim,
                'sttc_sim'        :   self.sttc_sim,
                'correlation'     :   self.correlation,
                'SC_max'          :   self.SC_max,
                'SC_dt_max'       :   self.SC_dt_max,
                'sttc_clustered'  :   self.sttc_clustered,
                'cluster_labels'  :   self.cluster_labels,
                'cluster_centers' :   self.cluster_centers
                }
            
        elif mode == 'complete':
            results_dict = {
                'mode'            :   mode,
                'pressure'        :   self.pressure,
                'file_ID'         :   self.file_ID,
                'N_units'         :   self.N_units,
                'N_samples'       :   self.N_samples,
                'T_total'         :   self.T_total,
                'sample_rate'     :   self.sample_rate,
                'locs'            :   self.locs,
                'trains'          :   self.trains,
                'trains_binary'   :   self.trains_binary,
                'firings_total'   :   self.firings_total,
                'rate_total'      :   self.rate_total,
                'time_rate_total' :   self.time_rate_total,
                'rates'           :   self.rates,
                'time_rates'      :   self.time_rates,
                'rates_avg'       :   self.rates_avg,
                'sttc'            :   self.sttc,
                'sttc_dt'         :   self.sttc_dt,
                'sim'             :   self.sim,
                'sttc_sim'        :   self.sttc_sim,
                'correlation'     :   self.correlation,
                'SC'              :   self.SC,
                'contrast'        :   self.contrast,
                'activeST'        :   self.activeST,
                'SC_dts'          :   self.SC_dts,
                'SC_max'          :   self.SC_max,
                'SC_dt_max'       :   self.SC_dt_max,
                'sttc_clustered'  :   self.sttc_clustered,
                'cluster_labels'  :   self.cluster_labels,
                'cluster_centers' :   self.cluster_centers
                }
            
        elif mode == 'minimal':
            results_dict = {
                'mode'            :   mode,
                'pressure'        :   self.pressure,
                'file_ID'         :   self.file_ID,
                'N_units'         :   self.N_units,
                'N_samples'       :   self.N_samples,
                'T_total'         :   self.T_total,
                'sample_rate'     :   self.sample_rate,
                'rate_total'      :   self.rate_total,
                'time_rate_total' :   self.time_rate_total,
                'correlation'     :   self.correlation,
                'SC_max'          :   self.SC_max,
                'SC_dt_max'       :   self.SC_dt_max
                }
        
        return results_dict
            
            
    def save_data(self, path : str = None,
                  run_all : bool = False,
                  mode : Literal['condensed', 'complete', 'minimal'] = 'complete') -> None:
        
        if path is None:
            path = f'results_{self.file_ID}.pkl'

        if run_all:
            self.run_all()
        
        results_dict = self.convert_results_to_dict(mode=mode)

        with open(path, 'wb') as file:
            pickle.dump(results_dict, file)

