# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import glob
import numpy as np
import pickle

from MEAData import MEAData



def pipe(pressures,
         input_path : str,
         output_path : str = 'results.pkl',
         extension : str = '\\sorted.npz') -> None:
    # get all file paths
    folders = glob.glob(input_path)

    files = []

    for folder in folders:
        files.append(folder + extension)

    # config
    # smoothing_params
    rate_total_window_size = 0.03
    rate_sigma = 0.05

    # sttc params
    dt_maxs = np.array([.04])
    dt_mins = np.array([0])

    # spike contrast params
    dt_range = (0.001, 1)

    # loop over all files for each pressure, compute wanted quantities and save them to a npz file
    results = []

    for pressure, file_path in zip(pressures, files):
        print('+' + '-'*24 + '+')
        print(f'|  FILE ID:  {file_path.split('\\')[-2]}'.ljust(23) + '  |')
        print(f'|  PRESSURE: {pressure} psi'.ljust(23) + '  |')
        print('+' + '-'*24 + '+')
        
        mea_data = MEAData(file_path, file_type='raw', overview=True, pressure=pressure)

        print('Calculating firing rate metrics...')
        mea_data.get_firings_total()
        mea_data.get_rate_total(window_size=rate_total_window_size)
        mea_data.get_rates(method='ISI', sigma=rate_sigma)
        mea_data.get_rates_avg()
        print()

        print(f'Calculating STTCs with dt_max = {dt_maxs} and dt_min = {dt_mins}...')
        mea_data.compute_sttc(dt_maxs=dt_maxs, dt_mins=dt_mins, use_numba=False)
        print()

        print(f'Calculating STTC to random comparison with dt_max = {dt_maxs} and dt_min = {dt_mins}......')
        mea_data.compare_STTC_to_random(plot=False, method='all')
        print()

        print('Calculating spike contrast...')
        mea_data.spike_contrast(dt_min=dt_range[0], dt_max=dt_range[1])
        print()

        print('Writing data to dictionary...')
        results.append(mea_data.convert_results_to_dict(mode='minimal'))
        print()

        print('Analysis complete')
        print()

    with open(output_path, 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    base_paths = ['C:\\Users\\bow-lab\\Documents\\Code\\data\\ABAB_3\\25327\\']
    trials = ['A_1', 'A_2', 'B_1', 'B_2']

    pressures = np.array([[0.13, 11.49, 17.33, 22.11, 24.90],
                          [0.15, 10.16, 15.61, 20.40, 25.20],
                          [0.12, 11.24, 15.30, 20.27, 24.48],
                          [0.16, 10.93, 15.27, 20.44, 24.13]])
    
    press_err = np.array([[.01, .08, .13, .13, .04],
                          [.06, .07, .14, .07, .03],
                          [.03, .05, .05, .03, .07],
                          [.07, .03, .07, .03, .03]])

    for base_path in base_paths:
        for i, trial in enumerate(trials):
            input_path = base_path + trial + '\\chunk*'
            output_path = f'results_{base_path.split('\\')[-2]}_{trial}.pkl'

            print('+' + '-'*24 + '+')
            print(f'|  MEA CHIP:  {base_path.split('\\')[-2]}'.ljust(23) + '  |')
            print(f'|  TRIAL: {trial}'.ljust(23) + '  |')
            print('+' + '-'*24 + '+')
            
            pipe(pressures[i], input_path, output_path)




