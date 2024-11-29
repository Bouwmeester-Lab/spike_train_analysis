# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import glob
from MEAData import MEAData
import numpy as np


# get all file paths
folders = glob.glob('C:\\Users\\bow-lab\\Documents\\Code\\data\\240917_results\\24039_results\\000*')

files = []

for folder in folders:
    files.append(f'{folder}\\sorted.npz')


pressures = np.array([2.14, 4.44, 6.54, 8.56, 10.18, 11.75, 16.02, 18.03, 20.12,
                      0,    2.17, 4.29, 4.29, 6.33, 9.09, 10.16, 14.09, 16.12, 18.00, 20.0])

# config
# smoothing_params
rate_total_window_size = 0.03
rate_sigma = 0.05

# sttc params
dt_max = 0.05
dt_min = 0

# spike contrast params
dt_range = (0.001, 1)

# loop over all files for each pressure, compute wanted quantities and save them to a npz file
results = []

for pressure, file_path in zip(pressures, files):
    print(f'___{file_path.split('\\')[-2]}___{pressure}___')
    mea_data = MEAData(file_path, file_type='raw', overview=False, pressure=pressure)

    print('Calculating firing rate metrics...')
    mea_data.get_firings_total()
    mea_data.get_rate_total(window_size=rate_total_window_size)
    mea_data.get_rates(sigma=rate_sigma)
    mea_data.get_rates_avg()

    print('Calculating STTC...')
    mea_data.compute_sttc(dt_max=dt_max, dt_min=0)
    print('Calculating clustered STTC...')
    mea_data.get_STTC_clustered()
    print('Calculating STTC to random comparison...')
    mea_data.compare_STTC_to_random(dt_max=dt_max, dt_min=dt_min, plot=False, method='all')

    print('Calculating spike contrast...')
    mea_data.spike_contrast(dt_min=dt_range[0], dt_max=dt_range[1])

    print('Writing data to dictionary...')
    results.append(mea_data.convert_results_to_dict(mode='complete'))

    print('Analysis complete')

np.savez('.\\results.npz', data=np.array(results, dtype=object))