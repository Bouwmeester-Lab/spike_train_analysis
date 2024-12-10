# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import numpy as np
import matplotlib.pyplot as plt
import spike_train_tools as stt
import pickle

from typing import Tuple, Literal, List, Union, Any
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
            self.pressure = pressure

            self.data = np.load(path, allow_pickle=True)

            self.units = self.data['units']
            self.locations = self.data['locations']
            self.sample_rate = int(self.data['fs'])
            self.remove_empty_trains()

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
            self.trains_sorted = None

            with open(path, 'rb') as file:
                self.data = pickle.load(file)

            self.parse_processed_data()

             
        else:
            print('file type must be either \'raw\' or \'processed\'')


        self.unit_numbers = np.arange(0, self.N_units)


        if overview:
            self.print_data_overview()


    
    def remove_empty_trains(self) -> None:
        del_indices = []
        for i, unit in enumerate(self.units):
            if len(unit['spike_train']) == 0:
                del_indices.append(i)

        self.units = np.delete(self.units, del_indices)
        self.locations = np.delete(self.locations, del_indices)



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
        self.correlation_error = None
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
        self.correlation_error = self.data['correlation_error']
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




    def print_data_overview(self, spacing : int = 18,
                            value_length : int = 140) -> None:
        '''
        Prints an overview of the MEA data. 
        '''

        # PRINT HEADER
        print('KEY'.ljust(spacing),
              'SHAPE'.ljust(spacing),
              'TYPE'.ljust(spacing),
              'VALUE')
        
        keys = list(self.data.keys())

        for key in keys:
            self._print_line(self.data, key, spacing, value_length)


        if self.file_type == 'raw':
            print()
            print('Each entry of the array of units is a dictionary:', end='\n\n')

            unit0 = self.data['units'][0]
            units_keys = unit0.keys()

            for key in units_keys:
                self._print_line(unit0, key, spacing, value_length)

        print()       
        print(f'Total measurement time: {int(self.T_total//60)} min {round(self.T_total%60, 3)} s', end='\n\n')



    def _print_line(self, data : Any,
                    key : str,
                    spacing : int,
                    value_length : int) -> None:
        
        content = data[key]
        content_shape = self._get_content_shape(content)
        content_type = self._get_content_type(content)

        print(key.ljust(spacing),
              content_shape.ljust(spacing),
              content_type.ljust(spacing),
              repr(content).replace('\n', '').ljust(value_length)[:value_length])


    
    def _get_content_type(self, content : Any) -> str:
        content_type = type(content).__name__
        if content_type == 'list' or content_type == 'ndarray':
            try:
                content_type += f'[{str(type(content[0]).__name__)}]'
            except:
                pass
        return content_type




    def _get_content_shape(self, content : Any) -> str:
        try:
            content_shape = str(np.shape(content))
        except ValueError:
            content_shape = 'inhomogeneous'
        return content_shape




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
    



    def raster_plot(self, sort : bool = True,
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

        stt.raster_plot(plot_trains, self.sample_rate, plot_avg, 
                        self.time_rate_total, self.rate_total, time_range)




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




    def compute_sttc(self, dt_max : Union[float, NDArray] = 0.05,
                     dt_min : Union[float, NDArray] = 0.,
                     plot : bool = False,
                     sort : bool = True,
                     tiling_method : Literal['fft', 'direct'] = 'direct',
                     use_numba : bool = False) -> NDArray:
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

        if type(dt_max) == float and type(dt_min) == float:
            self.sttc = stt.STTC(self.trains_binary, self.sample_rate,
                             dt_max=dt_max, dt_min=dt_min, plot=plot,
                             sort=sort, tiling_method=tiling_method,
                             use_numba=use_numba)
            self.sttc_dt = (dt_min, dt_max)

        else:
            if len(dt_max) == len(dt_min):
                self.sttc = []
                self.sttc_dt = []
                for dtmin, dtmax in zip(dt_min, dt_max):
                    self.sttc.append(stt.STTC(self.trains_binary, self.sample_rate,
                                              dt_max=dtmax, dt_min=dtmin, plot=plot,
                                              sort=sort, tiling_method=tiling_method,
                                              use_numba=use_numba))
                    self.sttc_dt.append((dtmin, dtmax))
            else:
                print('dt_mins and dt_maxs do not have the same size')
                return
            
        return self.sttc, self.sttc_dt




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
        sttc_result_ordered = self.sttc[-1][order][:,order]
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
        if ( self.sttc_clustered is None ) or ( int(np.shape(self.sttc_clustered)[0]) != int(len(self.cluster_centers)) ):
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




    def compute_sttc_sim(self, dt_max : float = 0.05,
                         dt_min : float = 0,
                         use_numba : bool = False):
        
        if self.rates_avg is None:
            self.get_rates_avg()

        self.sim = stt.simulate_spiketrains(self.N_units, self.rates_avg, self.T_total, self.sample_rate)
        
        self.sttc_sim = stt.STTC(self.sim, sample_rate=self.sample_rate, dt_max=dt_max, dt_min=dt_min, plot=False, use_numba=use_numba)

        return self.sim, self.sttc_sim
    


    def get_pdf(self, data : NDArray,
                bins : Union[int, str]) -> Tuple[NDArray, NDArray, float]:
        
        counts, bin_edges = np.histogram(data, bins=bins)

        bin_size = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[1:] - bin_size / 2

        total_counts = np.sum(counts)

        counts_norm = counts / total_counts / bin_size

        return counts_norm, bin_centers, bin_size
    

    
    def get_correlation(self, method : Literal['pdf_difference',
                                               'Cramer-von Mises',
                                               'Wasserstein',
                                               'all'],
                        pdf : NDArray = None, 
                        pdf_sim : NDArray = None, 
                        cdf : NDArray = None, 
                        cdf_sim : NDArray = None) -> NDArray:
        
        correlation = []

        if method == 'pdf_difference' or method == 'all':
            assert pdf is not None and pdf_sim is not None, 'pdf_difference method needs pdf and pdf_sim arguments'
            slice = (pdf_sim > pdf)
            correlation.append(np.sum(np.abs(pdf_sim[slice] - pdf[slice])))

        if method == 'Cramer-von Mises' or method == 'all':
            assert cdf is not None and cdf_sim is not None, 'Cramer-von Mises method needs cdf and cdf_sim arguments'
            correlation.append(np.sum((np.abs(cdf - cdf_sim)**2)))

        if method == 'Wasserstein' or method == 'all':
            assert cdf is not None and cdf_sim is not None, 'Wasserstein method needs cdf and cdf_sim arguments'
            correlation.append(np.sum(np.abs(cdf - cdf_sim)))

        correlation = np.array(correlation)
        
        return correlation


    def upper_triangle(self, data : NDArray) -> NDArray:
        return data[np.triu_indices(np.shape(data)[0], k=1)]


    
    def compare_STTC_to_random(self, bins_data : Union[int, str],
                               bins_sim : Union[int, str],
                               plot : bool = True,
                               method : Literal['pdf_difference',
                                                'Cramer-von Mises',
                                                'Wasserstein',
                                                'all'] = 'Wasserstein',
                               n_splits : int = 1,
                               plot_sim_hist : bool = False,
                               use_numba : bool = False) -> NDArray:
    
        if self.sttc is None:
            print('Compute STTC on data with dt_max = 0.05...')
            sttcs = self.compute_sttc()
        elif type(self.sttc) != list:
            sttcs = [self.sttc]
            dts = [self.sttc_dt]
        else:
            sttcs = self.sttc
            dts = self.sttc_dt

        dt_mins = [dt[0] for dt in dts]
        dt_maxs = [dt[1] for dt in dts]
 
        for i, (dt_min, dt_max, sttc) in enumerate(zip(dt_mins, dt_maxs, sttcs)):
            data = self.upper_triangle(sttc)

            _, sttc_sim = self.compute_sttc_sim(dt_max, dt_min, use_numba=use_numba)
            sim = self.upper_triangle(sttc_sim)
            pdf_sim, bin_centers_sim, bin_size_sim = self.get_pdf(sim, bins_sim)
            cdf_sim = np.cumsum(pdf_sim) * bin_size_sim
            popt, pcov = curve_fit(stt._gauss_norm, bin_centers_sim, pdf_sim, p0=[np.mean(pdf_sim), np.std(pdf_sim)])
            err = np.sqrt(np.diag(pcov))

            np.random.shuffle(data)
            splits = np.array_split(data, n_splits)

            for j, split in enumerate(splits):
                pdf, bin_centers, bin_size = self.get_pdf(split, bins_data)
                cdf = np.cumsum(pdf) * bin_size
        
                if plot:
                    self.plot_sttc_histogram(bin_centers, bin_centers_sim, popt, err,
                                             pdf, cdf, pdf_sim, cdf_sim, plot_sim_hist)
                
                corr_j = self.get_correlation(method, pdf, stt._gauss_norm(bin_centers, *popt),
                                                    cdf, stt._cum_gauss(bin_centers, *popt))
                print(corr_j)
                if j == 0:
                    corr_i = np.array([corr_j])
                else:
                    corr_i = np.vstack((corr_i, corr_j))

            if n_splits > 1:
                corr_mean = np.mean(corr_i, axis=0)
                corr_err = np.std(corr_i, axis=0)
            else:
                corr_mean = corr_i
                corr_err = np.full_like(corr_i, None)

            if i == 0:
                self.correlation = corr_mean
                self.correlation_error = corr_err
            else:
                self.correlation = np.vstack((self.correlation, corr_mean))
                self.correlation_error = np.vstack((self.correlation_error, corr_err))
   
        return self.correlation, self.correlation_error

 
    def plot_sttc_histogram(self, bin_centers_data : NDArray,
                            bin_centers_sim : NDArray,
                            popt : NDArray,
                            err : NDArray,
                            counts_norm : NDArray,
                            counts_cum : NDArray,
                            counts_sim_norm : NDArray,
                            counts_sim_cum : NDArray,
                            plot_sim_hist : bool) -> None:
        
        bin_size_data = np.mean(np.diff(bin_centers_data))
        bin_min_data, bin_max_data = bin_centers_data[0]-bin_size_data/2, bin_centers_data[-1]+bin_size_data/2

        bin_size_sim = np.mean(np.diff(bin_centers_sim))
        bin_min_sim, bin_max_sim = bin_centers_sim[0]-bin_size_sim/2, bin_centers_sim[-1]+bin_size_sim/2

        bin_continuous = np.linspace(min((bin_min_data, bin_min_sim)), max((bin_max_data, bin_max_sim)), 1000)

        gauss_cont = stt._gauss_norm(bin_continuous, *popt)

        cum_gauss_cont = stt._cum_gauss(bin_continuous, *popt)
        cum_gauss = stt._cum_gauss(bin_centers_data, *popt)

        _, axs = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle(f'Xenon pressure: {self.pressure} psi')

        if plot_sim_hist:
            axs[0].bar(bin_centers_sim, counts_sim_norm, width=bin_size_sim, color='white', edgecolor='k', alpha=.5, label='Poisson')
            axs[1].bar(bin_centers_sim, counts_sim_cum, width=bin_size_sim, color='white', edgecolor='k', alpha=.5, label='random sim')

        axs[0].set_title('pdf')
        axs[0].bar(bin_centers_data, counts_norm, width=bin_size_data, color='r', edgecolor='k', alpha=.5, label='data')
        axs[0].plot(bin_continuous, gauss_cont, c='k', linestyle='--', label='fit')
        axs[0].fill_between(bin_centers_data, stt._gauss_norm(bin_centers_data, *popt), counts_norm, where=(counts_norm > stt._gauss_norm(bin_centers_data, *popt)), 
                color='b', alpha=.3, label='correlation')
        axs[0].set_ylabel('Probability density')
        axs[0].text(.96*bin_max_data, .7 / np.sqrt(2*np.pi*popt[1]**2),
                    f'µ = {popt[0]:.4f} ± {err[0]:.4f}\nσ = {popt[1]:.4f} ± {err[1]:.4f}', 
                    horizontalalignment='right', verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        axs[0].legend()

        axs[1].set_title('cdf')
        axs[1].fill_between(bin_centers_data, counts_cum, cum_gauss, where=(cum_gauss > counts_cum), 
                color='b', alpha=0.3)
        axs[1].fill_between(bin_centers_data, counts_cum, cum_gauss, where=(cum_gauss <= counts_cum), 
                color='b', alpha=0.3)
        axs[1].bar(bin_centers_data, counts_cum, width=bin_size_data, color='r', edgecolor='k', alpha=.5, label='data')
        axs[1].plot(bin_continuous, cum_gauss_cont, c='k', linestyle='--', label='fit')
        axs[1].set_ylabel('Cumulative probability density')
        

        for ax in axs:
            ax.set_xlabel('STTC [-]')
            ax.set_xlim(bin_min_data, bin_max_data)
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
            self.compute_sttc_sim()
        if self.sttc_sim is None or self.correlation is None:
            self.compare_STTC_to_random(plot=False, method='all')


    def convert_results_to_dict(self,
                                mode : Literal['condensed', 'complete', 'minimal']) -> dict:
        if mode == 'condensed':
            results_dict = {
                'mode'             :   mode,
                'pressure'         :   self.pressure,
                'file_ID'          :   self.file_ID,
                'N_units'          :   self.N_units,
                'N_samples'        :   self.N_samples,
                'T_total'          :   self.T_total,
                'sample_rate'      :   self.sample_rate,
                'locs'             :   self.locs,
                'trains'           :   self.trains,
                'sttc'             :   self.sttc,
                'sttc_dt'          :   self.sttc_dt,
                'sim'              :   self.sim,
                'sttc_sim'         :   self.sttc_sim,
                'correlation'      :   self.correlation,
                'correlation_error':   self.correlation_error,
                'SC_max'           :   self.SC_max,
                'SC_dt_max'        :   self.SC_dt_max,
                'sttc_clustered'   :   self.sttc_clustered,
                'cluster_labels'   :   self.cluster_labels,
                'cluster_centers'  :   self.cluster_centers
                }
            
        elif mode == 'complete':
            results_dict = {
                'mode'             :   mode,
                'pressure'         :   self.pressure,
                'file_ID'          :   self.file_ID,
                'N_units'          :   self.N_units,
                'N_samples'        :   self.N_samples,
                'T_total'          :   self.T_total,
                'sample_rate'      :   self.sample_rate,
                'locs'             :   self.locs,
                'trains'           :   self.trains,
                'trains_binary'    :   self.trains_binary,
                'firings_total'    :   self.firings_total,
                'rate_total'       :   self.rate_total,
                'time_rate_total'  :   self.time_rate_total,
                'rates'            :   self.rates,
                'time_rates'       :   self.time_rates,
                'rates_avg'        :   self.rates_avg,
                'sttc'             :   self.sttc,
                'sttc_dt'          :   self.sttc_dt,
                'sim'              :   self.sim,
                'sttc_sim'         :   self.sttc_sim,
                'correlation'      :   self.correlation,
                'SC'               :   self.SC,
                'contrast'         :   self.contrast,
                'activeST'         :   self.activeST,
                'SC_dts'           :   self.SC_dts,
                'SC_max'           :   self.SC_max,
                'SC_dt_max'        :   self.SC_dt_max,
                'sttc_clustered'   :   self.sttc_clustered,
                'cluster_labels'   :   self.cluster_labels,
                'cluster_centers'  :   self.cluster_centers
                }
            
        elif mode == 'minimal':
            results_dict = {
                'mode'             :   mode,
                'pressure'         :   self.pressure,
                'file_ID'          :   self.file_ID,
                'N_units'          :   self.N_units,
                'N_samples'        :   self.N_samples,
                'T_total'          :   self.T_total,
                'sample_rate'      :   self.sample_rate,
                'rate_total'       :   self.rate_total,
                'time_rate_total'  :   self.time_rate_total,
                'correlation'      :   self.correlation,
                'correlation_error':   self.correlation_error,
                'SC_max'           :   self.SC_max,
                'SC_dt_max'        :   self.SC_dt_max
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

