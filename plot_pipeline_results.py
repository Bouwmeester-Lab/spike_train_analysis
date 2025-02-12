import matplotlib.pyplot as plt
import pickle
import numpy as np
from numpy.typing import NDArray
from typing import Union, Literal

plt.rcdefaults()
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')

def summarize_pipeline_results(files : Union[str, list],
                               runs : Union[str, list] = None,
                               correlation_plot : Literal['pdf_difference','Cramer-Von Mises','Wasserstein'] = 'Wasserstein'):
    
    if runs == None:
        runs = []
        for file in files:
            runs.append(file.split('\\')[-1])
    
    _, axs = plt.subplots(6, 2, figsize=(18, 24), sharex=True)

    colors = ['r', 'g', 'b', 'orange']
    markers = ['s']*4
    capsize = 7
    ecolor = 'k'
    elinewidth = None

    for i, (file, run, c, m) in enumerate(zip(files, runs, colors, markers)):
        with open(file, 'rb') as file:
            data = pickle.load(file)
            pressures = []
            pressure_err = []
            N_units = []
            T_total = []
            N_firings = []
            rates = np.array([])
            corr = np.array([])
            corr_err = np.array([])
            sttc_dts = []
            sc = []
            sc_err = []
            dt = []
            dt_err = []

            for i, d in enumerate(data):
                pressures.append(d['pressure'])
                pressure_err.append(d['pressure_err'])
                N_units.append(d['N_units'])
                T_total.append(d['T_total'])
                N_firings.append(d['N_firings'])
                rates = np.append(rates, np.mean(d['rate_total']))

                corr = np.append(corr, d['correlation'][0][correlation_plot])
                corr_err = np.append(corr_err, d['correlation'][0][correlation_plot+'_err'])
                print(d['correlation'][0][correlation_plot], d['sttc_dt'])
                sttc_dts.append(d['sttc_dt'])

                sc.append(d['SC_max'])
                sc_err.append(d['SC_max_err'])
                dt.append(d['SC_dt_max'])
                dt_err.append(d['SC_dt_max_err'])


        axs[0,0].set_ylabel('# neurons [-]')           
        axs[0,0].errorbar(pressures, N_units, xerr=pressure_err,
                          label=run, color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)

        axs[0,1].set_ylabel('Total measurement time [s]')
        axs[0,1].errorbar(pressures, T_total, xerr=pressure_err,
                          label=run, color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)


        axs[1,0].set_ylabel('Total number of firings [-]')
        axs[1,0].errorbar(pressures, N_firings, xerr=pressure_err,
                          label=run, color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)
        
        axs[1,1].set_ylabel('Norm. total number of firings [-]')        
        axs[1,1].errorbar(pressures, N_firings/N_firings[0], xerr=pressure_err,
                          label=run, color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)
        

        axs[2,0].set_ylabel('Average firing rate [Hz]')        
        axs[2,0].errorbar(pressures, rates, xerr=pressure_err,
                          label=run, color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)
        
        axs[2,1].set_ylabel('Norm. average firing rate [-]')        
        axs[2,1].errorbar(pressures, rates/rates[0], xerr=pressure_err,
                          label=run, color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)
        
        print(sttc_dts)
        print(corr, corr_err)
        axs[3,0].set_ylabel(f'{correlation_plot} correlation [-]')      
        axs[3,0].errorbar(pressures, corr, xerr=pressure_err, yerr=corr_err,
                          color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)
        
        axs[3,1].set_ylabel(f'Norm. {correlation_plot} correlation [-]')        
        axs[3,1].errorbar(pressures, corr/corr[0], xerr=pressure_err, yerr=corr_err/corr[0],
                          color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)


        axs[4,0].set_ylabel('Spike contrast [-]')        
        axs[4,0].errorbar(pressures, sc, xerr=pressure_err, yerr=sc_err,
                          label=run, color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)
        
        axs[4,1].set_ylabel('Norm. spike contrast [-]')        
        axs[4,1].errorbar(pressures, sc/sc[0], xerr=pressure_err, yerr=sc_err/sc[0],
                          label=run, color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)
        

        axs[5,0].set_ylabel(r'$\Delta t$ spike contrast [s]')        
        axs[5,0].errorbar(pressures, dt, xerr=pressure_err, yerr=dt_err,
                          label=run, color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)
        
        axs[5,1].set_ylabel(r'Norm. $\Delta t$ spike contrast [-]')        
        axs[5,1].errorbar(pressures, dt/dt[0], xerr=pressure_err, yerr=dt_err/dt[0],
                          label=run, color=c, marker=m, capsize=capsize,
                          ecolor=ecolor, elinewidth=elinewidth)

        for row in axs:
            for ax in row:
                ax.set_xlabel('Pressure [psi]')
                ax.legend()
                ax.grid()

    plt.show()


def summarize_pipeline_results2(files : Union[str, list],
                               runs : Union[str, list] = None,
                               correlation_plot : Literal['pdf_difference','Cramer-Von Mises','Wasserstein'] = 'Wasserstein',
                               savefile : str = 'C:\\Users\\bow-lab\\Documents\\Code\\figures\\pipeline_results.pdf'):
    
    if runs == None:
        runs = []
        for file in files:
            runs.append(file.split('\\')[-1])
    
    _, axs = plt.subplots(3, 2, figsize=(18, 18), sharex=True)
    plt.subplots_adjust(wspace=0.18, hspace=0.03)
    plt.suptitle('Effects of Xe-129 and Xe-130 isotopes on organoids', fontsize=20, y=.92)

    colors = ['r', 'g', 'b', 'orange']
    markers = ['s', '^', 's', '^']
    capsize = 0
    ecolor = colors#'k'
    elinewidth = None

    for i, (file, run, c, m) in enumerate(zip(files, runs, colors, markers)):
        with open(file, 'rb') as file:
            data = pickle.load(file)
            pressures = []
            pressure_err = []
            N_units = []
            T_total = []
            N_firings = []
            rates = np.array([])
            corr = np.array([])
            corr_err = np.array([])
            sttc_dts = []
            sc = []
            sc_err = []
            dt = []
            dt_err = []

            for i, d in enumerate(data):
                pressures.append(0.06894757*d['pressure'])
                pressure_err.append(0.06894757*d['pressure_err'])
                N_units.append(d['N_units'])
                T_total.append(d['T_total'])
                N_firings.append(d['N_firings'])
                rates = np.append(rates, np.mean(d['rate_total']))

                corr = np.append(corr, d['correlation'][0][correlation_plot])
                corr_err = np.append(corr_err, d['correlation'][0][correlation_plot+'_err'])
                sttc_dts.append(d['sttc_dt'])

                sc.append(d['SC_max'])
                sc_err.append(d['SC_max_err'])
                dt.append(d['SC_dt_max'])
                dt_err.append(d['SC_dt_max_err'])


        axs[0,0].set_ylabel('Number of units', fontsize=14)           
        axs[0,0].errorbar(pressures, N_units, xerr=pressure_err,
                          label=run, color=c, marker=m, elinewidth=elinewidth)


        axs[0,1].set_ylabel('Total number of firings [-]', fontsize=14)
        axs[0,1].errorbar(pressures, N_firings, xerr=pressure_err,
                          label=run, color=c, marker=m, elinewidth=elinewidth)
        

        axs[1,0].set_ylabel('Average population rate [Hz]', fontsize=14)
        axs[1,0].errorbar(pressures, rates, xerr=pressure_err,
                          label=run, color=c, marker=m, elinewidth=elinewidth)
        
        axs[1,1].set_ylabel(f'Correlation [-]', fontsize=14)
        axs[1,1].errorbar(pressures, corr, xerr=pressure_err, yerr=corr_err,
                          color=c, marker=m, elinewidth=elinewidth)
        print(corr_err)

        axs[2,0].set_ylabel('Spike contrast [-]', fontsize=14)
        axs[2,0].errorbar(pressures, sc, xerr=pressure_err, yerr=sc_err,
                          label=run, color=c, marker=m, elinewidth=elinewidth)
        axs[2,0].set_xlabel('Pressure [bar]', fontsize=14)
        
        axs[2,1].set_ylabel(r'$\Delta t_{SC}$ [s]', fontsize=14)
        axs[2,1].errorbar(pressures, dt, xerr=pressure_err, yerr=dt_err,
                          label=run, color=c, marker=m, elinewidth=elinewidth)
        axs[2,1].set_xlabel('Pressure [bar]', fontsize=14)
    
    

        for row in axs:
            for ax in row:
                ax.grid(True)

        ax.legend(ncols=4, bbox_to_anchor=(.6, -0.215), fontsize=14)

    #plt.tight_layout()
    plt.savefig(savefile)
    plt.show()



if __name__ == '__main__':
    import glob

    folders = ['C:\\Users\\bow-lab\\Documents\\Code\\results\\ABAB\\AB_organoid\\results_MEA\\results_*.pkl']
    
    for folder in folders:
        files = glob.glob(folder)
    
        print('files found:')
        for file in files:
            print(file)

        #files = [files[0], files[2], files[1], files[3]]

        runs = ['A', 'B']

        summarize_pipeline_results2(files, runs, 'Cramer-Von Mises', savefile=f'C:\\Users\\bow-lab\\Documents\\Code\\figures\\pipeline_{folder.split('\\')[-2]}.pdf')
