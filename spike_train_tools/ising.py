# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Tuple, Union, List
from numpy.typing import NDArray
from tqdm import tqdm
from numba import njit, jit

plt.rcdefaults()
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')

@njit 
def activity(trains : NDArray) -> NDArray:
    '''Computes the average activity (magnetization) of a set of neural spins.

    Params:
    - trains     (NDArray) : set of neural spins states (spike trains) of size (N_neurons, N_samples)

    Returns:
    - activities (NDArray) : array of average magnetizations of size (N_neurons)
    '''
    activities = np.average(trains, axis=1)
    return activities


@njit
def covariance(trains : NDArray) -> NDArray:
    '''Computes the covariance of a set of neural spins.

    Params:
    - trains      (NDArray) : set of neural spins states (spike trains) of size (N_neurons, N_samples)

    Returns:
    - covariances (NDArray) : array of covariances of size (N_neurons, N_neurons)
    '''
    size = np.shape(trains)[0]
    covariances = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j and i > j:
                cov = np.mean(trains[i]*trains[j]) - np.mean(trains[i]) * np.mean(trains[j])
                covariances[i,j] = covariances[j,i] = cov
    return covariances


@jit
def ising_hamiltonian(state : NDArray,
                      h : NDArray,
                      J : NDArray) -> float:
    '''Computes the total energy of a neural spin state. A state refers to the all the neural spins
    at a single moment in time

    Params:
    - state     (NDArray) : neural spin state of size (N_neurons)
    - h         (NDArray) : proportial term in Hamiltonian of size (N_neurons)
    - J         (NDArray) : interaction term in Hamiltonian of size (N_neurons, N_neurons)          

    Returns:
    - H_tot     (float)   : Total energy of the state
    '''
    H_tot = -1*np.sum(h*state)
    N_neurons = int(len(state))
    for i in range(N_neurons):
        for j in range(i+1, N_neurons):
            H_tot -= J[i, j] * state[i] * state[j]
    return H_tot


@jit
def metropolis_step(state : NDArray,
                    energy : float,
                    h : NDArray,
                    J : NDArray,
                    T : float = 1) -> Tuple[NDArray, float]:
    '''Performs a single step in the metropolis algorithm

    Params:
    - state     (NDArray) : neural spin state of size (N_neurons)
    - energy    (float)   : energy of the state
    - h         (NDArray) : proportial term in Hamiltonian of size (N_neurons)
    - J         (NDArray) : interaction term in Hamiltonian of size (N_neurons, N_neurons)
    - T         (NDArray) : temperature. default is normalized with T = 1   

    Returns:
    - state_new     (NDArray) : neural spin state of size (N_neurons)
    - energy_new    (float)   : energy of the state
    '''
    # pick a random firing event
    N_neurons = int(len(state))
    i = np.random.randint(0, N_neurons)

    # flip a 0 to a 1 and vice versa
    state_new = np.copy(state)
    state_new[i] = -1*state[i]

    #calculate the change in energy due to the flip
    energy = ising_hamiltonian(state, h, J)
    energy_new = ising_hamiltonian(state_new, h, J)
    dE = energy_new - energy

    if dE < 0:
        return state_new, energy_new
    elif dE >= 0:
        a = np.random.random()
        if a < np.exp(-1*dE / T):
            return state_new, energy_new
        else:
            return state, energy
        


@jit
def run_metropolis(state : NDArray,
                   N_iterations : int,
                   h : NDArray,
                   J : NDArray,
                   T : float = 1) -> Tuple[NDArray, NDArray]:
    '''Runs a simulation of an ising model using the Metropolis algorithm for N_iterations

    Params:
    - state         (NDArray) : initial neural spin state of size (N_neurons)
    - N_iterations  (int)     : energy of the state
    - h             (NDArray) : proportial term in Hamiltonian of size (N_neurons)
    - J             (NDArray) : interaction term in Hamiltonian of size (N_neurons, N_neurons)
    - T             (NDArray) : temperature. default is normalized with T = 1   

    Returns:
    - states        (NDArray) : set of neural spin state of size (N_neurons, N_iterations)
    - energies      (NDArray) : energies of the simulated states of size (N_iterations)
    '''
    N_neurons = int(len(state))
    states = np.zeros((N_neurons, N_iterations))
    states[:,0] = state

    energies = np.zeros(N_iterations)
    energy = ising_hamiltonian(state, h, J)
    energies[0] = energy

    
    for iteration in range(1, N_iterations):
        state, energy = metropolis_step(state, energy, h, J, T)
        states[:,iteration] = state
        energies[iteration] = energy
    return states, energies


def plot_ising_overview(states : NDArray,
                        energies : NDArray) -> None:
    firings = ((states+1)/2).astype(bool)
    activity = np.sum(firings, axis=0)
    N_neurons, N_its = np.shape(states)
    units = np.arange(N_neurons)

    _, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, train in enumerate(firings):
        if i == 0:
            label = 'Firings'
        else:
            label = None

        firing_times = np.where(train)

        axs[0].scatter(firing_times, np.full_like(firing_times, units[i]),
                       s=.03, color='k', label=label, alpha=.03)

    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Units')
    axs[0].set_ylim(0, N_neurons)

    axs[1].plot(activity)
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Average magnetization')

    axs[2].plot(energies)
    axs[2].set_xlabel('Iterations')
    axs[2].set_ylabel('Total energy')

    for ax in axs:
        ax.set_xlim(0, N_its)

    plt.show()


def upper_triangle(a : NDArray) -> NDArray:
    return a[np.triu_indices(np.shape(a)[0], k=1)]



def fit_ising(trains_observations : NDArray,
              h : NDArray,
              J : NDArray,
              N_its_equilibrium : int,
              N_its_fit : int,
              state0 : NDArray = None,
              h_learning_rate : float = 2,
              h_learning_rate_scaling : float = -0.4,
              J_learning_rate : float = 1,
              J_learning_rate_scaling : float = -0.4,
              T : float = 1) -> Tuple[NDArray, NDArray, dict]:
    '''Runs a simulation of an ising model using the Metropolis algorithm for N_iterations

    Params:
    - trains_observations   (NDArray) : spike train observations to fit on of size (N_neurons, N_samples)
    - h             (NDArray) : proportial term in Hamiltonian of size (N_neurons)
    - J             (NDArray) : interaction term in Hamiltonian of size (N_neurons, N_neurons)
    - N_its_equilibrium (int) : number of iterations taken to equilibrize the ising simulation
    - N_its_fit     (NDArray) : 
    - state0        (NDArray) : the initial neural spin state that is used for the simulations, size (N_neurons)
                                default is [-1]*N_neurons, i.e. no active spins
    - learning_rate (str)     : learning rate
    - learning_rate_scaling (float) : learning rate scaling. If non-zero, the learning rate scales as learning_rate*n^(learning_rate_scaling)
    - T             (NDArray) : temperature. default is normalized with T = 1   

    Returns:
    - h             (NDArray) : fitted values for h of size (N_neurons)
    - J             (NDArray) : fitted values for J of size (N_neurons, N_neurons)
    - info          (dict)    : additional info on h and J. 
    '''
    N_neurons, N_its = np.shape(trains_observations)
    if state0 is None:
        state0 = -1*np.ones(N_neurons)

    activity_obs = activity(trains_observations)
    cov_obs = covariance(trains_observations)

    h_means = [np.mean(h)]
    J_means = [np.mean(J[np.triu_indices(np.shape(J)[0], k=1)])]
    h_stds = [np.std(h)]
    J_stds = [np.std(J[np.triu_indices(np.shape(J)[0], k=1)])]
    activity_errors = []
    covariance_errors = []

    for n in tqdm(range(1, N_its_fit)):
        states, _ = run_metropolis(state0, N_its_equilibrium + N_its, h, J, T)
        trains = states[:,-1*int(N_its):]

        h_eta = h_learning_rate*n**(h_learning_rate_scaling)
        J_eta = J_learning_rate*n**(J_learning_rate_scaling)

        activity_MC = activity(trains)
        covariance_MC = covariance(trains)
        
        h -= h_eta*(activity_MC - activity_obs)
        J -= J_eta*(covariance_MC - cov_obs)

        h_means.append(np.mean(h))
        J_means.append(np.mean(J[np.triu_indices(np.shape(J)[0], k=1)]))
        h_stds.append(np.std(h))
        J_stds.append(np.std(J[np.triu_indices(np.shape(J)[0], k=1)]))
        activity_errors.append(np.mean(activity_MC - activity_obs))
        covariance_errors.append(np.mean(upper_triangle(covariance_MC - cov_obs)))
        #print(f'activity avg err: {np.mean(activity_MC - activity_obs)}')
        #print(f'covariance avg err: {np.mean(upper_triangle(covariance_MC - cov_obs))}')
        
    info = {'h_mean'   :   h_means,
            'h_std'    :   h_stds,
            'J_mean'   :   J_means,
            'J_std'    :   J_stds,
            'act_err'  :   activity_errors,
            'cov_err'  :   covariance_errors}

    return h, J, info