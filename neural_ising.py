from spike_train_tools.ising import run_metropolis, plot_ising_overview, activity, covariance, fit_ising, plot_fit_results
from spike_train_tools.tools import simulate_spiketrains

import numpy as np
import matplotlib.pyplot as plt


N_its_eq = 2000
N_its_sim = 1000
N_its_fit = 10000


### import data #####################
from MEAData import MEAData

file_path = 'C:\\Users\\bow-lab\\Documents\\Code\\data\\ABAB\\ABAB_3\\25286\\A_1\\chunk0\\sorted.npz'

mea_data = MEAData(file_path, file_type='raw', pressure=0.0, overview=False)

trains = mea_data.convert_trains_to_binary()[:,:2000000]

N_neurons = mea_data.N_units

### average data ###
dt = 0.02
dN = int(dt * mea_data.sample_rate)

steps = int(np.shape(trains)[1] / dN) - 1

states = np.zeros((N_neurons, steps))

for i, train in enumerate(trains):
    steps = int(len(train) / dN) - 1
    train_avg = np.zeros(steps)
    for j in range(steps):
        train_avg[j] = np.clip(np.sum(train[j*dN:(j+1)*dN]), 0, 1)
    states[i] = 2*train_avg - 1
        

### fit Ising ###
activity_obs = activity(states)
cov_obs = covariance(states)

h0 = 1*np.ones(N_neurons)
J0 = -1*np.ones((N_neurons, N_neurons))
J0[np.diag_indices(N_neurons)] = 0
state0 = 1*np.ones(N_neurons)

h, J, info = fit_ising(states, h0, J0, N_its_eq, N_its_fit, state0)


plot_fit_results(h, J, info)
