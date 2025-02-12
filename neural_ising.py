from spike_train_tools.ising import fit_ising, activity, covariance
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

trains_obs = np.loadtxt('C:\\Users\\bow-lab\\Documents\\Code\\data\\Sampaio et. al\\sample1.dat', delimiter=',').T

N_neurons, N_samples = np.shape(trains_obs)

N_its_eq = 1000
N_its_fit = 60000

activity_obs = activity(trains_obs)
cov_obs = covariance(trains_obs)

# make initial guesses
h = np.zeros(N_neurons)
J = np.zeros((N_neurons, N_neurons))
state0 = -1*np.ones(N_neurons)

h, J, info = fit_ising(trains_obs, h, J, N_its_eq, N_its_fit, state0,
                           h_learning_rate=.002, h_learning_rate_scaling=0,
                           J_learning_rate=.002, J_learning_rate_scaling=0)