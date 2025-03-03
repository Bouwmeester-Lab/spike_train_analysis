import numpy as np
import os
from spike_train_tools.tools import simulate_spiketrains
from spike_train_tools.ising import run_metropolis, activity, covariance
from MEAData import MEAData

option = 'data' #'data', 'simulation'

N_neurons = 4
rate = 50
T_total = 100
sample_rate = 1000

if option == 'poisson':
    binary_trains = simulate_spiketrains(N_neurons, rate, T_total, sample_rate)

elif option == 'simulate':
    state = -1*np.ones(N_neurons)
    N_iterations = 2000
    h = 1*np.ones(N_neurons)
    J = -1*np.ones((N_neurons, N_neurons))
    J[np.diag_indices(N_neurons)] = 0
    
    states, energies = run_metropolis(state, N_iterations, h, J)
    binary_trains = states[:,500:]

elif option  == 'data':
    path = 'C:\\Users\\bow-lab\\Documents\\Code\\data\\ABAB\\ABAB_3\\25286\\A_1\\chunk0\\sorted.npz'

    meadata = MEAData(path, 'raw', pressure=0, pressure_err=0)

    binary_trains = meadata.convert_trains_to_binary()
    N_neurons = meadata.N_units


magnetizations = np.mean(binary_trains, axis=1)
p11 = np.zeros(int(N_neurons*(N_neurons)/2))
p_11 = np.zeros(int(N_neurons*(N_neurons)/2))
p1_1 = np.zeros(int(N_neurons*(N_neurons)/2))
p_1_1 = np.zeros(int(N_neurons*(N_neurons)/2))

k=0
for i in range(N_neurons):
    for j in range(N_neurons):
        if i != j and j>i:
            p_1_1[k] = np.mean((binary_trains[i]-1)*(binary_trains[j]-1))
            p_11[k] = np.mean(-1*(binary_trains[i]-1)*binary_trains[j])
            p1_1[k] = np.mean(-1*binary_trains[i]*(binary_trains[j]-1))
            p11[k] = np.mean(binary_trains[i]*binary_trains[j])
            k += 1

print(binary_trains[i],binary_trains[j],(binary_trains[i]-1)*(binary_trains[j]-1),np.mean((binary_trains[i]-1)*(binary_trains[j]-1)))

write_file_path = '.\\correlations.p'

with open(write_file_path, 'w') as file:
    for m in magnetizations:
        file.write(f'{1-m}\t{m}\n')
    for l in range(k):
        file.write(f'{p_1_1[l]}\t{p_11[l]}\t{p1_1[l]}\t{p11[l]}\n')








