# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import glob
import os
import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List
import pickle
import json

from MEAData import MEAData



def pipe(pressures : NDArray,
         press_err : NDArray,
         input_files : str,
         output_file : str,
         config_file : str) -> None:

    with open(config_file, "r") as file:
        config = json.load(file)

    print('Configuration:')
    print(json.dumps(config, indent=4))
    
    results = []

    for pressure, err, input_file in zip(pressures, press_err, input_files):
        print('+' + '-'*30 + '+')
        print(f'|  CHUNK:  {input_file.split('\\')[-2]}'.ljust(29) + '  |')
        print(f'|  PRESSURE: {pressure} ± {err} psi'.ljust(29) + '  |')
        print('+' + '-'*30 + '+')
        
        # make a MEAData instance
        data = MEAData(input_file, file_type='raw', overview=config['general']['overview'], pressure=pressure, pressure_err=err)

        print('Calculating firing rate metrics...')
        data.get_firings_total()
        data.get_rate_total(window_size=config['get_rate_total']['window_size'])
        data.get_rates(method=config['get_rates']['method'],
                       sigma=config['get_rates']['sigma'],
                       plot=False)
        data.get_rates_avg()
        print()

        print(f'Calculating STTCs with dt_max = {config['compute_sttc']['dt_max']} and dt_min = {config['compute_sttc']['dt_min']}...')
        data.compute_sttc(dt_max=config['compute_sttc']['dt_max'],
                          dt_min=config['compute_sttc']['dt_min'],
                          tiling_method=config['compute_sttc']['tiling_method'],
                          use_numba=config['compute_sttc']['use_numba'],
                          plot=False)
        print()

        print(f'Calculating STTC to random comparison with dt_max = {config['compute_sttc']['dt_max']} and dt_min = {config['compute_sttc']['dt_min']}......')
        data.compare_sttc_to_random(bins_data=config['compare_sttc_to_random']['bins_data'],
                                    bins_sim=config['compare_sttc_to_random']['bins_sim'],
                                    n_splits=config['compare_sttc_to_random']['n_splits'],
                                    min_samples_per_hist=config['compare_sttc_to_random']['min_samples_per_hist'],
                                    use_numba=config['compare_sttc_to_random']['use_numba'],
                                    plot=False)
        print()

        print('Calculating spike contrast...')
        data.spike_contrast(dt_min=config['spike_contrast']['dt_min'],
                            dt_max=config['spike_contrast']['dt_max'],
                            N_points=config['spike_contrast']['N_points'],
                            stride_bin_ratio=config['spike_contrast']['stride_bin_ratio'],
                            err_estimate=config['spike_contrast']['err_estimate'],
                            err_fitrange=config['spike_contrast']['err_fitrange'],
                            err_fitdegree=config['spike_contrast']['err_fitdegree'],
                            plot=False, err_plot=False)
        print()

        print('Writing data to dictionary...')
        results.append(data.convert_results_to_dict(mode=config['general']['saving_mode']))
        print()

        print('Analysis complete')
        print()

    make_folder(output_file)

    with open(output_file, 'wb') as file:
        pickle.dump(results, file)


def make_folder(path : str) -> None:
    folder_tree = path.split('\\')
    current_folder = folder_tree[0]
    for folder in folder_tree[1:-1]:
        current_folder += '\\' + folder
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)


def print_header(measurement_id : str,
                 chip_ID : str,
                 trial : str,
                 padding : int = 40) -> None:
    line1 = f'MEAS ID: {measurement_id}'
    line2 = f'CHIP ID:  {chip_ID}'
    line3 = f'TRIAL: {trial}'
    max_len = np.max([len(line1), len(line2), len(line3)])
    total_length = max_len + 2*padding

    print('#'*total_length)
    print('#'*(padding-1) + f' {line1.ljust(max_len)} ' + '#'*(padding-1))
    print('#'*(padding-1) + f' {line2.ljust(max_len)} ' + '#'*(padding-1))
    print('#'*(padding-1) + f' {line3.ljust(max_len)} ' + '#'*(padding-1))
    print('#'*total_length)



def parse_pressure_log(file_path : str,
                       trials : List[str] = ['A_1', 'B_1', 'A_2', 'B_2']) -> dict:
    
    pressures = {}
    
    current_trial = {'index'  : [],
                     'median' : [],
                     'mean'   : [],
                     'std'    : []}
    
    trial_counter = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith('='):
                # Finalize the current section when encountering a separator
                if current_trial['index']:
                    trial_key = trials[trial_counter]  # Convert counter to letter (A, B, C...)
                    pressures[trial_key] = current_trial
                    current_trial = {'index'  : [],
                                     'median' : [],
                                     'mean'   : [],
                                     'std'    : []}
                    trial_counter += 1

            elif line.startswith('Interval') and not line.startswith('Interval Index'):
                # Parse data lines
                parts = line.split('\t')
                parts = [p.strip() for p in parts if p.strip()]  # Clean and filter parts
                current_trial['index'].append(parts[0])
                current_trial['median'].append(float(parts[1]))
                current_trial['mean'].append(float(parts[2]))
                current_trial['std'].append(float(parts[3]))


    # Add the last section if it exists
    if current_trial['index']:
        trial_key = trials[trial_counter]
        pressures[trial_key] = current_trial

    return pressures


def main() -> None:
    base_folder = 'C:\\Users\\bow-lab\\Documents\\Code\\data\\ABAB'
    config_file = 'C:\\Users\\bow-lab\\Documents\\Code\\spike_train_analysis\\config.json'
    measurements = [{'measurement_ID'  : 'AB_organoid',
                     'chip_IDs'         : ['MEA'],
                     'trials'           : ['A_1', 'B_1']}]

    for measurement in measurements:
        id = measurement['measurement_ID']

        pressures = parse_pressure_log(f'{base_folder}\\{id}\\pressures.txt', trials=measurement['trials'])

        for chip_id in measurement['chip_IDs']:
            for trial in measurement['trials']:
                print_header(id, chip_id, trial)

                input_folders = f'{base_folder}\\{id}\\{chip_id}\\{trial}\\chunk*'
                print(input_folders)
                input_files = glob.glob(input_folders)
                input_files = [file + '\\sorted.npz' for file in input_files]

                print('input files:')
                for file in input_files:
                    print(file)

                output_file = f'C:\\Users\\bow-lab\\Documents\\Code\\results\\ABAB\\{id}\\results_{chip_id}\\results_{trial}.pkl'
                print(f'output_file:\n {output_file}')

                pipe(pressures[trial]['mean'], pressures[trial]['std'],
                     input_files, output_file, config_file)



if __name__ == '__main__':
    main()
   


    
    




