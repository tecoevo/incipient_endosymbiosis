import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from matplotlib.lines import Line2D
import multiprocessing as mp

verbose = True

# to store evolutionary trajectories

allrunsH = []
allrunsS = []

# parameters that ensure stability

CH = 100.0
CS = 2*CH # we assume that the carrying capacity of the symbiont is twice that of the host
CHS = 5*CH

rH = 8.0
rS = 20.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
rHS = 10.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.

a = 0.1 
d0 = 50.0

# initialise lists that will hold values of the obligacies - these will be plotted against each other later to view the trajectory

timesteps = 1000
no_runs = 1

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

mutation_std = 0.0001

def fH(omegaH):
     return rH*(1-omegaH)

def fS(omegaS):
     return rS*(1-omegaS)

def fHS(omegaH, omegaS, alphaH, alphaS):
     return rHS*omegaH*omegaS*alphaH*alphaS

def d(alphaH, alphaS):
     return d0*(1-alphaH*alphaS)

def GetPopDynSoln(traitH, traitS):
     omegaH = traitH[0]
     omegaS = traitS[0]
     alphaH = traitH[1]
     alphaS = traitS[1]

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
         xH += dt*(fH(omegaH)*xH*(1-xH/CH) - a*xH*xS + d(alphaH, alphaS)*xHS)
         xS += dt*(fS(omegaH)*xS*(1-xS/CS) - a*xH*xS + d(alphaH, alphaS)*xHS)
         xHS += dt*(fHS(omegaH, omegaS, alphaH, alphaS)*xHS*(1-xHS/CHS) + a*xH*xS - d(alphaH, alphaS)*xHS)

         if time%100000 == 0: # check every so often if the routine has converged
             if xH - np.mean(xHarray[-1:-10:-1])<0.001 and \
             xS - np.mean(xSarray[-1:-10:-1])<0.001 and \
             xHS - np.mean(xHSarray[-1:-10:-1])<0.001:
                 return [xH, xS, xHS]

         elif time>steps:
             return print("Maximum time exceeded. Perhaps the fixed point doesn't exist/isn't stable.")
         
def MutateHost(currentH, currentS, xHstar, xSstar, xHSstar, trajectoryH, trajectoryS):
     # induce mutation in host trait and decide fate of the mutant. 
     trajectoryS.append(currentS)
     mutantH = np.clip(np.random.multivariate_normal(currentH, [[mutation_std, 0], [0, mutation_std]])
     ,a_min=0,a_max=1)
     mutantH_jacobian = [[fH(mutantH[0])*(1-xHstar/CH) - a*xSstar, d(mutantH[1],currentS[1])], 
     [a*xSstar, fHS(mutantH[0], currentS[0], mutantH[1], currentS[1])*(1-xHSstar/CHS) - d(mutantH[1], currentS[1])]]
     mutantH_eigenvalues = np.linalg.eigvals(mutantH_jacobian)
     if np.max([np.real(val) for val in mutantH_eigenvalues])>0:
         trajectoryH.append(mutantH)
     else:
         trajectoryH.append(currentH)

def MutateSymbiont(currentH, currentS, xHstar, xSstar, xHSstar, trajectoryH, trajectoryS):
     # induce mutation in symbiont trait and decide fate of the mutant.
     trajectoryH.append(currentH)
     mutantS = np.clip(np.random.multivariate_normal(currentS, [[mutation_std, 0], [0, mutation_std]])
     ,a_min=0,a_max=1)
     mutantS_jacobian = [[fS(mutantS[0])*(1-xSstar/CS) - a*xHstar, d(currentH[1],mutantS[1])], 
     [a*xHstar, fHS(currentH[0], mutantS[0], currentH[1], mutantS[1])*(1-xHSstar/CHS) - d(currentH[1], mutantS[1])]]
     mutantS_eigenvalues = np.linalg.eigvals(mutantS_jacobian)
     if np.max([np.real(val) for val in mutantS_eigenvalues])>0:
         trajectoryS.append(mutantS)
     else:
         trajectoryS.append(currentS)

def run_model(run_no):
     # Runs one simulation of the lattice model and saves the trajectory of traits of H and S at each mutation
     # time point. 
     # returns a list [run number, trait_trajectoryH, trait_trajectoryS]
     
     if verbose==True:
         print("started run number", run_no)
     trait_trajectoryH = [[0.001, 0.001]]
     trait_trajectoryS = [[0.001, 0.001]]
     for t in range(timesteps):
         if t%20==0:
             print("now at evolution timestep ",t)
         currentH = trait_trajectoryH[-1]
         currentS = trait_trajectoryS[-1]

         # calculate fixed point of population dynamics
         try:
             xHstar, xSstar, xHSstar = GetPopDynSoln(currentH, currentS)
         except:
             print("ode solver didn't converge. timesteps: ", len(trait_trajectoryH))
             print("trait value at this time. traitH = ", currentH, "traitS = ", currentS)
             break
         # is it feasible?
         if xHstar<0 or xSstar<0 or xHSstar<0:
             if verbose==True:
                 print("Fixed point became infeasible. Number of timesteps completed: ", len(trait_trajectoryH))
             break

         # is it stable? since we are numerically solving this, if the routine found the fixed point,
         # then it is necessarily stable. We may perform a sanity check by checking the eigenvalues of 
         # resident Jacobian as below:
         #  jacobian = [[fH(omegaH)*(1-2*xHstar/CH) - a*xSstar, -a*xHstar, d(alphaH, alphaS)],
         #              [-a*xSstar, fS(omegaS)*(1-2*xSstar/CS) - a*xHstar, d(alphaH, alphaS)],
         #              [a*xSstar, a*xHstar, fHS(omegaH, omegaS, alphaH, alphaS)*(1-2*xHSstar/CHS)  - d(alphaH, alphaS)]]
         
         # for stability, we use the eigenvalues: the fixed point of the dynamical system giving
         # rise to the above Jacobian is stable iff all its eigenvalues have negative real part
         #  resident_eigenvalues = np.linalg.eigvals(jacobian)
     
         # if any of the oligacies are 1, for example the host, then the host cannot live independently.
         # therefore, the population goes to zero abundance and no further mutants can arise.
         # strictly speaking, the below block isn't necessary because np.clip when generating the mutant trait
         # value should keep the trait at 1 no matter how many times Mutate is called. But removing this would 
         # lead to numerous useless Mutate calls and wasted time.

         if any(np.array(currentH) >= 1.0) and any(np.array(currentS) >= 1.0):
             break
         if any(np.array(currentH) >= 1.0) >= 1.0:
             MutateSymbiont(currentH,currentS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
             continue
         if any(np.array(currentS) >= 1) >= 1.0:
             MutateHost(currentH,currentS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
             continue
 
         # start the exponential clocks for host and symbiont populations
         clocks = [random.expovariate(xHstar), random.expovariate(xSstar)]
         if np.min(clocks) == clocks[0]:
             # the host clock went off first and so now we mutate the host trait
             MutateHost(currentH,currentS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
         if np.min(clocks) == clocks[1]:
             # the host clock went off first and so now we mutate the host trait
             MutateSymbiont(currentH,currentS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
         elif clocks[0]==clocks[1]:
             # mutate host with probability 1/2, and symbiont with probability 1/2
             mutantpicker = random.uniform(0.0,1.0)
             if mutantpicker<0.5:
                 MutateHost(currentH,currentS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
             elif 0.5<=mutantpicker<=1:
                 MutateSymbiont(currentH,currentS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)

     return [run_no, trait_trajectoryH, trait_trajectoryS]

def record_result(result):
     # required for the "callback" arg in the multiprocessing.apply_async function to record the result given by each CPU 
     _, resultH, resultS = result
     global allrunsH
     global allrunsS
     allrunsH.append(resultH)
     allrunsS.append(resultS)

if __name__ == '__main__':
     allrunsH = []
     allrunsS = []

     pool = mp.Pool(no_runs) # number of CPUs

     for run in range(1,no_runs+1):
         #  print('run number ',run, 'started out of ', no_runs)
         pool.apply_async(run_model, args=[run], callback=record_result)
        
     pool.close() # close the processes
     pool.join()  # finalise the processes by recording results

     # to plot stochastic obligacy trajectories
     for run in range(no_runs):
         traitH = np.array(allrunsH[run])
         traitS = np.array(allrunsS[run])

         omegaH_traj = np.around(traitH[:,0], 4)
         omegaS_traj= np.around(traitS[:,0], 4)
         dependence  = omegaH_traj*omegaS_traj

         alphaH_traj = np.around(traitH[:,1], 4)
         alphaS_traj = np.around(traitS[:,1], 4)
         cohesion = alphaH_traj*alphaS_traj

         data=pd.DataFrame(list(zip(omegaH_traj, alphaH_traj, omegaS_traj, alphaS_traj)), columns=['omegaH', 'alphaH','omegaS', 'alphaS'])
         data.to_csv("results/data/coevolution-trajectory_rH="+str(rH) + "_rS=" + str(rS) +"_rHS=" + str(rHS)+"_d0="+str(d0)+"_run"+str(run)+".txt", index=False)
         plt.plot(omegaH_traj, omegaS_traj, label='omegas')
         plt.plot(alphaH_traj, alphaS_traj, label='alphas')
         plt.plot(dependence,cohesion, label='both')

     plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
     plt.xlim(0,1.1)
     plt.ylim(0,1.1)
     plt.grid(visible=True, axis='both')
     plt.gca().set_aspect('equal')
     plt.legend()
     plt.show()
     plt.savefig("coevolution_rH="+str(rH) + "_rS=" + str(rS) +"_rHS=" + str(rHS)  +".pdf",format='pdf')


