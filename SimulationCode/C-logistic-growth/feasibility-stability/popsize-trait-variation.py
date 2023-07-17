import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.integrate import odeint 
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.lines import Line2D

which = 'sigma'

# parameters that seem reasonable

CH = 100.0
CS = 2*CH # we assume that the carrying capacity of the symbiont is twice that of the host
CHS = 5*CS
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.
a = 0.1 

# initialise lists that will hold values of the obligacies - these will be plotted against each other later to view the trajectory

timesteps = 1000
no_runs = 100

if which=='omega':
    rH = 8.0
    rS = 20.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
    rHS = 10.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
    d = 50.0

    omegas = np.arange(0,1.00, 0.05)
    omega_arraylength = len(omegas)
    xHstar = np.zeros([omega_arraylength, omega_arraylength])
    xSstar = np.zeros([omega_arraylength,omega_arraylength])
    xHSstar = np.zeros([omega_arraylength,omega_arraylength])

    for i in range(omega_arraylength):
        omegaH = omegas[i]
        print("starting omegaH = ", omegaH) 
        for j in range(omega_arraylength):
            omegaS = omegas[j]         
            steps = 10000000
            
            dt = 0.001 # timestep
            # initial conditions
            xH = 10.0
            xS = 10.0
            xHS = 0.0
            xHarray = []
            xSarray = []
            xHSarray = []
            for t in range(1,steps+1):
                xHarray.append(xH)
                xSarray.append(xS)
                xHSarray.append(xHS)
                xH += dt*(rH*(1-omegaH)*xH*(1-xH/CH) - a*xH*xS + d*xHS)
                xS += dt*(rS*(1-omegaS)*xS*(1-xS/CS)- a*xH*xS + d*xHS)
                xHS += dt*(rHS*omegaH*omegaS*xHS*(1-xHS/CHS) + a*xH*xS - d*xHS)

                if t%100000 == 0: # check every so often if the routine has converged
                    if xH - np.mean(xHarray[-1:-10:-1])<0.001 and \
                    xS - np.mean(xSarray[-1:-10:-1])<0.001 and \
                    xHS - np.mean(xHSarray[-1:-10:-1])<0.001:
                        xHstar[omega_arraylength-1-j][i] = xH
                        xSstar[omega_arraylength-1-j][i] =  xS 
                        xHSstar[omega_arraylength-1-j][i] =  xHS
                        break

                if t==steps:
                    xHstar[omega_arraylength-1-j][i] = 0
                    xSstar[omega_arraylength-1-j][i] = 0
                    xHSstar[omega_arraylength-1-j][i] = 0

    minvalue = round(np.min([np.max(xHstar), np.min(xSstar), np.min(xHSstar)]), 2)
    maxvalue = round(np.max([np.max(xHstar), np.max(xSstar), np.max(xHSstar)]), 2)

    labels = ['']*len(omegas)

    for i in range(0, len(omegas)-1, 2):
        labels[i] = round(omegas[i], 2)

    print("min eq pop size. abort if any of these are negative - this means that somewhere, the numerical routine \
    is relaxing onto a fixed point where population size is negative.", np.min(xHstar), np.min(xSstar), np.min(xHSstar))

    sns.heatmap(xHstar, vmin=minvalue, vmax=maxvalue, xticklabels = labels, yticklabels = np.flip(labels), cbar_kws = {'ticks':np.around(np.linspace(minvalue, maxvalue, 7),0)})
    plt.savefig("host-pop-size-variation-omegas.pdf", format='pdf')
    plt.show()

    sns.heatmap(xSstar, vmin=minvalue, vmax=maxvalue, xticklabels = labels, yticklabels = np.flip(labels), cbar_kws = {'ticks':np.around(np.linspace(minvalue, maxvalue, 7),0)})
    plt.savefig("symbiont-pop-size-variation-omegas.pdf", format='pdf')
    plt.show()

    sns.heatmap(xHSstar, vmin=minvalue, vmax=maxvalue, xticklabels = labels, yticklabels = np.flip(labels), cbar_kws = {'ticks':np.around(np.linspace(minvalue, maxvalue, 7),0)})
    plt.savefig("collective-pop-size-variation-omegas.pdf", format='pdf')
    plt.show()

    sns.heatmap(xHstar-xSstar, vmin=-maxvalue, vmax=maxvalue, xticklabels = labels, yticklabels = np.flip(labels), cbar_kws = {'ticks':np.around(np.linspace(-maxvalue, maxvalue, 7),0)}) 
    plt.savefig("hostminussymbiont-pop-size.pdf", format='pdf')
    plt.show()

if which=='sigma':
    fH = 8.0
    fS = 20.0
    rHS = 10.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
    d0 = 50.0

    sigmas = np.arange(0,1.00, 0.05)
    sigma_arraylength = len(sigmas)
    xHstar = np.zeros([sigma_arraylength, sigma_arraylength])
    xSstar = np.zeros([sigma_arraylength,sigma_arraylength])
    xHSstar = np.zeros([sigma_arraylength,sigma_arraylength])

    for i in range(sigma_arraylength):
        sigmaH = sigmas[i]
        print("starting omegaH = ", sigmaH) 
        for j in range(sigma_arraylength):
            sigmaS = sigmas[j]         
            steps = 10000000
            
            dt = 0.001 # timestep
            # initial conditions
            xH = 10.0
            xS = 10.0
            xHS = 0.0
            xHarray = []
            xSarray = []
            xHSarray = []
            for t in range(1,steps+1):
                xHarray.append(xH)
                xSarray.append(xS)
                xHSarray.append(xHS)
                xH += dt*(fH*xH*(1-xH/CH) - a*xH*xS + d0*(1-sigmaH*sigmaS)*xHS)
                xS += dt*(fS*xS*(1-xS/CS) - a*xH*xS + d0*(1-sigmaH*sigmaS)*xHS)
                xHS += dt*(rHS*sigmaH*sigmaS*xHS*(1-xHS/CHS) + a*xH*xS - d0*(1-sigmaH*sigmaS)*xHS)

                if t%100000 == 0: # check every so often if the routine has converged
                    if xH - np.mean(xHarray[-1:-10:-1])<0.001 and \
                    xS - np.mean(xSarray[-1:-10:-1])<0.001 and \
                    xHS - np.mean(xHSarray[-1:-10:-1])<0.001:
                        xHstar[sigma_arraylength-1-j][i] = xH
                        xSstar[sigma_arraylength-1-j][i] =  xS 
                        xHSstar[sigma_arraylength-1-j][i] =  xHS
                        break

                if t==steps:
                    xHstar[sigma_arraylength-1-j][i] = 0
                    xSstar[sigma_arraylength-1-j][i] = 0
                    xHSstar[sigma_arraylength-1-j][i] = 0

    minvalue = round(np.min([np.max(xHstar), np.min(xSstar), np.min(xHSstar)]), 2)
    maxvalue = round(np.max([np.max(xHstar), np.max(xSstar), np.max(xHSstar)]), 2)

    labels = ['']*sigma_arraylength

    for i in range(0, sigma_arraylength-1, 2):
        labels[i] = round(sigmas[i], 2)

    print("min eq pop size. abort if any of these are negative - this means that somewhere, the numerical routine \
    is relaxing onto a fixed point where population size is negative.", np.min(xHstar), np.min(xSstar), np.min(xHSstar))

    sns.heatmap(xHstar, vmin=minvalue, vmax=maxvalue, xticklabels = labels, yticklabels = np.flip(labels), cbar_kws = {'ticks':np.around(np.linspace(minvalue, maxvalue, 7),0)})
    plt.savefig("host-pop-size-variation-sigmas.pdf", format='pdf')
    plt.show()

    sns.heatmap(xSstar, vmin=minvalue, vmax=maxvalue, xticklabels = labels, yticklabels = np.flip(labels), cbar_kws = {'ticks':np.around(np.linspace(minvalue, maxvalue, 7),0)})
    plt.savefig("symbiont-pop-size-variation-sigmas.pdf", format='pdf')
    plt.show()

    sns.heatmap(xHSstar, vmin=minvalue, vmax=maxvalue, xticklabels = labels, yticklabels = np.flip(labels), cbar_kws = {'ticks':np.around(np.linspace(minvalue, maxvalue, 7),0)})
    plt.savefig("collective-pop-size-variation-sigmas.pdf", format='pdf')
    plt.show()

    sns.heatmap(xHstar-xSstar, vmin=-maxvalue, vmax=maxvalue, xticklabels = labels, yticklabels = np.flip(labels), cbar_kws = {'ticks':np.around(np.linspace(-maxvalue, maxvalue, 7),0)}) 
    plt.savefig("hostminussymbiont-pop-size-sigmas.pdf", format='pdf')
    plt.show()

