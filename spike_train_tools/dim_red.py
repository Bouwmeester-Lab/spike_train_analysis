# Code written by G.L. Timmerman
# Contact: timmerman@ucsb.edu

import numpy as np
import matplotlib.pyplot as plt
try:
    import umap.umap_ as umap
except:
    print('WARNING: no UMAP available')

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Literal
from numpy.typing import NDArray

plt.rcdefaults()
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')


def dim_red(rates : NDArray,
            dimension : int,
            method : Literal['PCA', 'tSNE', 'umap'] = 'PCA',
            plot : bool = True,
            ranges : NDArray = None) -> NDArray:
    '''
    Performs dimensional reduction on spike train data by mapping each unit (neuron)
    onto a separate dimension. 

    params:
    - firings_rates (NDArray)  :   array of shape (#neurons, #timestamps)
    - dimension     (int)       :   lower dimension to reduce data to 
    - method        (Literal)   :   dimension reduction algorithm to be used
                                        PCA: Principal Component Analysis
                                        t-SNE: T-Distributed sSochastic Neighbor Embedding
                                        UMAP: Uniform Manifold Approximation and Projection
    - plot          (bool)      :   determined whether results are plotted
    - ranges        (NDArray)  :   array with ranges for plot of size (dimension, 2)

    returns:    
    - components    (NDArray)  :   array of size (dimension, #timestamps)
    '''
    
    if method == 'PCA':
        model = PCA

    elif method == 'tSNE':
        model = TSNE

    elif method == 'umap':
        model = umap.UMAP
    
    # calculate reduced components
    components = model(n_components=dimension).fit_transform(rates.T)

    # plot
    if plot:
        fig = plt.figure()

        if dimension == 2:
            ax = fig.add_subplot(111)
            ax.scatter(components[:,0], components[:,1], s=5, alpha=.5, c='k')

        elif dimension == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(components[:,0], components[:,1], components[:,2], s=5, c='k', marker='o', alpha=.05)

            ax.set_zlabel('Component 3')
            if ranges is not None:
                ax.set_zlim(ranges[2][0], ranges[2][1])
                
        else:
            print('Dimension must be 2 or 3 for plotting')
        
        ax.set_title(f'{dimension}D {method} of neuronal firing data')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        if ranges is not None:
            ax.set_xlim(ranges[0][0], ranges[0][1])
            ax.set_ylim(ranges[1][0], ranges[1][1])

        plt.show()

    return components
