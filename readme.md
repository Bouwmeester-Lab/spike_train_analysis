# MEA spike train analysis

Code written by: G.L. Timmerman
contact: timmerman@ucsb.edu

This repository contains code used to perform analysis on neuronal spike trains, in particular data measured with MicroElectrode Arrays (MEAs). The code consists of two main parts: `spike_train_tools.py` and `MEAData.py`. 

### `spike_train_tools.py`
`spike_train_tools.py` contains several general functions that can be applied to spike trains, like dimensional reduction analysis, computing the Spike Time Tiling Coefficient (STTC), spike contrast, smoothing spike trains to obtain firing rates and methods to generate simulated data of spike trains. This module uses mainly binary numpy-array to store the spike trains, with each column referring to a measurement frame and each row corresponding with a neuron (also called unit). If a neuron fires in a certain frame, that entry of the binary matrix will become one. All function have elaborate docstrings, so please refer to the source code for more information on the different functions and their arguments. 

### `MEAData.py`
`MEAData.py` contains the `MEAData`-class. This class stores all data obtain from the MEAs. It takes a `.npz`-file that is outputted by Kilosort2. Kilosort2 is an algorithm that converts the MEA-data to spiketrains. The `MEAData`-class parses this output and contains several methods to perform analysis on this data, including computing the Spike Time Tiling Coefficient (STTC), spike contrast, smoothing spike trains to obtain firing rates and spacial clustering. All these methods use the `spike_train_tools`-module on the backend. The class also includes numerous plotting and data saving functions. All methods have elaborate docstrings, so please refer to the source code for more information on the different functions and their arguments. Please refer to `mea_data_tutorial.ipynb` for a hands-on explanation of how to use the `MEAData`-class. Example data for this tutorial can be downloaded from https://drive.google.com/file/d/1Z3yGWnMpBC43nHej2LLbZZjt9ht07mtl/view?usp=drive_link. `analysis_pipeline.py` runs all the analysis methods contained within the `MEAData`-class on multiple files (filepaths need to be defined within the script), and saves this data. This is useful for analizing bulk data. Note that performing all the analysis on one raw `.npz`-file might take over 30 minutes of computation time.



