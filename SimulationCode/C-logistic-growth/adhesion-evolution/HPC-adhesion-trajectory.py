import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.lines import Line2D
import multiprocessing as mp
import pandas as pd 
import argparse

if __name__ == "__main__":
     #this check makes sure that arguments are asked for only when this file is the main program($ python3 ...) and not when it is imported in another piece of code
     parser = argparse.ArgumentParser(description="Numerical simulation-based evolution experiment on the obligacies, where the host-endosymbiont collective has logistic growth.  ")

     parser.add_argument("-r", "--run", dest="run", help="run, type=int. the index of the run of this experiment.  ", required=True)

     parser.add_argument("-v", "--verb", dest="verbose", help="type=bool. do you want a verbose output?  ", default=False)
     
     parser.add_argument("-t", "--time", dest="timesteps", help="type=int. how many timesteps (mutation events) should the experiment run for?  ", default=3000)

     args = parser.parse_args()
     run = int(args.run)
     verbose = bool(args.verbose)
     timesteps = int(args.timesteps)

# to store evolutionary trajectories
# parameters that ensure stability

CH = 100.0
CS = 2*CH # we assume that the carrying capacity of the symbiont is twice that of the host
CHS = 5*CH

fH = 8.0
fS = 20.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
rHS = 40.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.

a = 0.1 
d0 = 50.0

# initialise lists that will hold values of the obligacies - these will be plotted against each other later to view the trajectory

# we model evolution of the trait (omega_i, alpha_i) as a continuous-time Markov chain (CTMC).
# Mutations can arise, when the H-S-HS population is at dynamical equilibrium, in either the host or symbiont. 
# Which of these it will first arrive in is decided using the result that the waiting time between jumps of a CTMC 
# are exponentially distributed with rate equal to the transition rate between the two states. Here we only care about 
# which of the two mutants - host or symbiont - arises first and not about the exact time it takes, so we will use their 
# equilibrium population abundances as proxies. This is motivated by the fact that the coefficient of the selection  
# gradient in the canonical equation is composed of three terms - the mutation rate, the variance of the mutation  
# distribution, and the equilibrium population abundance.  See the main text where we show that for this simple case, the 
# selection gradient is 1 for both. We assume that the mutational process for host and symbiont is similar, and so which 
# of the two obligacies grows faster is determined only by the equilibrium population abundance. 

mutation_std = 0.01

def d(alphaH, alphaS):
     return d0*(1-alphaH*alphaS)

def fHS(alphaH, alphaS):
     return rHS*alphaH*alphaS

def GetPopDynSoln(alphaH, alphaS):
     steps = 10000000
     
     dt = 0.001 # timestep
     # initial conditions
     xH = 10.0
     xS = 10.0
     xHS = 0.0
     xHarray = []
     xSarray = []
     xHSarray = []
     for time in range(1,steps+1):
         xHarray.append(xH)
         xSarray.append(xS)
         xHSarray.append(xHS)
         xH += dt*(fH*xH*(1-xH/CH) - a*xH*xS + d(alphaH, alphaS)*xHS)
         xS += dt*(fS*xS*(1-xS/CS)- a*xH*xS + d(alphaH, alphaS)*xHS)
         xHS += dt*(fHS(alphaH, alphaS)*xHS*(1-xHS/CHS) + a*xH*xS - d(alphaH, alphaS)*xHS)

         if time%100000 == 0: # check every so often if the routine has converged
             if xH - np.mean(xHarray[-1:-10:-1])<0.001 and \
             xS - np.mean(xSarray[-1:-10:-1])<0.001 and \
             xHS - np.mean(xHSarray[-1:-10:-1])<0.001:
                 return [xH, xS, xHS]

         elif time>steps:
             return print("Maximum time exceeded. Perhaps the fixed point doesn't exist/isn't stable.")

def MutateHost(currentH, currentS, xHstar, xSstar, xHSstar, trajectoryH, trajectoryS):
     # induce mutation in host obligacy and decide fate of the mutant. For derivation of
     # invasion criterion, see main text.
     # Briefly, mutants invade when they decrease the value of d/fHS
     trajectoryS.append(currentS)
     mutantH = np.clip(random.gauss(currentH,mutation_std), a_min=0, a_max=1)

     mutantH_jacobian = [[fH*(1-xHstar/CH) - a*xSstar, d(mutantH, currentS)], 
     [a*xSstar, fHS(mutantH, currentS)*(1-xHSstar/CHS) - d(mutantH, currentS)]]
     mutantH_eigenvalues = np.linalg.eigvals(mutantH_jacobian)
     realpartsH = [np.real(val) for val in mutantH_eigenvalues]
     if np.max(realpartsH)>10**(-7):
         trajectoryH.append(mutantH)
     else:
         trajectoryH.append(currentH)

def MutateSymbiont(currentH, currentS, xHstar, xSstar, xHSstar, trajectoryH, trajectoryS):
     # induce mutation in host obligacy and decide fate of the mutant. For derivation of
     # invasion criterion, see main text.
     # Briefly, mutants invade when they decrease the value of d/fHS
     trajectoryH.append(currentH)
     mutantS = np.clip(random.gauss(currentS,mutation_std), a_min=0, a_max=1)
     mutantS_jacobian = [[fS*(1-xSstar/CS) - a*xHstar, d(currentH, mutantS)], 
     [a*xHstar, fHS(currentH, mutantS)*(1-xHSstar/CHS) - d(currentH, mutantS)]]
     mutantS_eigenvalues = np.linalg.eigvals(mutantS_jacobian)
     if np.max([np.real(val) for val in mutantS_eigenvalues])>10**(-7):
         trajectoryS.append(mutantS)
     else:
         trajectoryS.append(currentS)
     
trait_trajectoryH = [0.001]
trait_trajectoryS = [0.001]

for t in range(timesteps):
    if verbose==True and t%20==0:
        print("now at evolution timestep ",t)
    alphaH = trait_trajectoryH[-1]
    alphaS = trait_trajectoryS[-1]
    print(alphaH, alphaS, '\n')

    # calculate fixed point of population dynamics
    try:
        xHstar, xSstar, xHSstar = GetPopDynSoln(alphaH, alphaS)
    except:
        print("ode solver didn't converge. timesteps: ", len(trait_trajectoryH))
        print("trait value at this time. traitH = ", alphaH, "traitS = ", alphaS)
        break
    # is it feasible?
    if xHstar<0 or xSstar<0 or xHSstar<0:
        if verbose==True:
            print("Fixed point became infeasible. Number of timesteps completed: ", len(trait_trajectoryH))
        break

    #  # is it stable? since we are numerically solving this, if the routine found the fixed point,
    #  # then it is necessarily stable. We may perform a sanity check by checking the eigenvalues of 
    #  # resident Jacobian as below:
    #  jacobian = [[fH(omegaH)*(1-2*xHstar/CH) - a*xSstar, -a*xHstar, d],
    #              [-a*xSstar, fS(omegaS)*(1-2*xSstar/CS) - a*xHstar, d],
    #              [a*xSstar, a*xHstar, fHS(omegaH, omegaS)*(1-2*xHSstar/CHS)  - d]]
    
    #  # for stability, we use the eigenvalues: the fixed point of the dynamical system giving
    #  # rise to the above Jacobian is stable iff all its eigenvalues have negative real part
    #  resident_eigenvalues = np.linalg.eigvals(jacobian)

    # if any of the oligacies are 1, for example the host, then the host cannot live independently.
    # therefore, the population goes to zero abundance and no further mutants can arise.
    # strictly speaking, the below block isn't necessary because np.clip when generating the mutant trait
    # value should keep the trait at 1 no matter how many times Mutate is called. But removing this would 
    # lead to numerous useless Mutate calls and wasted time.

    if alphaH >= 1.0:
        MutateSymbiont(alphaH, alphaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
        continue
    if alphaS >= 1.0:
        MutateHost(alphaH, alphaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
        continue
    if alphaH >= 1.0 and alphaS >= 1:
        break

    # start the exponential clocks for host and symbiont populations
    clocks = [random.expovariate(xHstar), random.expovariate(xSstar)]
    if np.min(clocks) == clocks[0]:
        # the host clock went off first and so now we mutate the host trait
        MutateHost(alphaH, alphaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
    if np.min(clocks) == clocks[1]:
        # the host clock went off first and so now we mutate the host trait
        MutateSymbiont(alphaH, alphaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
    elif clocks[0]==clocks[1]:
        # mutate host with probability 1/2, and symbiont with probability 1/2
        mutantpicker = random.uniform(0.0,1.0)
        if mutantpicker<0.5:
            MutateHost(alphaH, alphaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
        elif 0.5<=mutantpicker<=1:
            MutateSymbiont(alphaH, alphaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)

trait_trajectoryH = np.around(trait_trajectoryH, 4)
trait_trajectoryS = np.around(trait_trajectoryS, 4)

# save data for stickiness trajectories
data = pd.DataFrame(list(zip(trait_trajectoryH, trait_trajectoryS)), columns=['alphaH','alphaS'])
data.to_csv("results/data/stickiness-trajectory_rHS="+str(rHS) + "_d0=" + str(d0)+"_run"+str(run)+".txt", index=False) 

