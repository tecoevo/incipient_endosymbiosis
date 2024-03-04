import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from matplotlib.lines import Line2D
import multiprocessing as mp
import os

def HomogeniseLength(oldx,oldy, newlen):
     # given a pair of lists of equal length with entries from [0,1], output a list of pair of lists s.t.
     # the new pair of lists has a user-determined length (determined by the `step` argument).
     # new list is made by artificially extending the old list such that there are new copies of the last
     # elements of the old lists - as many as necessary to get them up to the required length.
     oldlen = len(oldx) # this should be equal to len(oldy)
     
     lastx = oldx[-1]
     lasty = oldy[-1]
     
     newx=list(oldx)
     newy=list(oldy)
     for i in range(newlen-oldlen):
         newx.append(lastx)
         newy.append(lasty)

     return (np.array(newx), np.array(newy))

verbose = True

# to store evolutionary trajectories
# parameters that ensure stability

CH = 100.0
CS = 2*CH # we assume that the carrying capacity of the symbiont is twice that of the host
CHS = 500.0

rH = 8.0
rS = 20.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
rHS = 40.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.

a = 5.0 
d = 50.0

jobname = 'a=' + str(a) + '_rHS=' + str(rHS) + '_CHS=' + str(CHS) # if there is further subdivision within the folder 'data'. if not, leave empty

# initialise lists that will hold values of the obligacies - these will be plotted against each other later to view the trajectory

timesteps = 1000 #1000
no_runs = 75 #75

# make sure all directories necessary to store results exist

try:
    os.mkdir("results/data/"+jobname)
except:
    pass

try:
    os.mkdir("results/plots/"+jobname)
except:
    pass

# we model evolution of the trait (omega_i, sigma_i) as a continuous-time Markov chain (CTMC).
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

def fH(omegaH):
     return rH*(1.-omegaH)

def fS(omegaS):
     return rS*(1.-omegaS)

def fHS(omegaH, omegaS):
     return rHS*omegaH*omegaS

def GetPopDynSoln(omegaH, omegaS):
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
         xH += dt*(rH*(1-omegaH)*xH*(1-xH/CH) - a*xH*xS + d*xHS)
         xS += dt*(rS*(1-omegaS)*xS*(1-xS/CS)- a*xH*xS + d*xHS)
         xHS += dt*(rHS*omegaH*omegaS*xHS*(1-xHS/CHS) + a*xH*xS - d*xHS)

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
     print('mutantH value: ', mutantH)

     host_inflow_dueto_encounterrateH = (d*a*xSstar)/(d - fHS(mutantH, currentS) + (fHS(mutantH, currentS)*xHSstar)/(CHS))
     host_inflow_dueto_independentreproduction = fH(mutantH)
     host_outflow = a*xSstar + (fH(mutantH)*xHstar)/(CH)
     R0H = (host_inflow_dueto_independentreproduction + host_inflow_dueto_encounterrateH)/host_outflow
     print("R0H = ", R0H)

     # can also do: 
    #  mutantH_jacobian = [[fH(mutantH)*(1-xHstar/CH) - a*xSstar, d], 
    #  [a*xSstar, fHS(mutantH, currentS)*(1-xHSstar/CHS) - d]]
    #  mutantH_eigenvalues = np.linalg.eigvals(mutantH_jacobian)
    #  realpartsH = [np.real(val) for val in mutantH_eigenvalues]
    #  print("mutantH eigvals", mutantH_eigenvalues)
    #  print("rounded realparts ", realpartsH)
     
     if R0H>1 + 10**(-7): # to exclude floating point errors
         trajectoryH.append(mutantH)
         print('mutantH accepted')
     else:
         trajectoryH.append(currentH)
         print('mutantH rejected')

def MutateSymbiont(currentH, currentS, xHstar, xSstar, xHSstar, trajectoryH, trajectoryS):
     # induce mutation in host obligacy and decide fate of the mutant. For derivation of
     # invasion criterion, see main text.
     # Briefly, mutants invade when they decrease the value of d/fHS
     trajectoryH.append(currentH)
     mutantS = np.clip(random.gauss(currentS,mutation_std), a_min=0, a_max=1)
     print('mutantS value: ', mutantS)

     symb_inflow_dueto_encounterrateH = (d*a*xHstar)/(d - fHS(currentH, mutantS) + (fHS(currentH, mutantS)*xHSstar)/(CHS))
     symb_inflow_dueto_independentreproduction = fS(mutantS)
     symb_outflow = a*xHstar + (fS(mutantS)*xSstar)/(CS)
     R0S = (symb_inflow_dueto_independentreproduction + symb_inflow_dueto_encounterrateH)/symb_outflow
     print("R0S = ", R0S)
    
     # can also do: 
    #  mutantS_jacobian = [[fS(mutantS)*(1-xSstar/CS) - a*xHstar, d], 
    #  [a*xHstar, fHS(currentH, mutantS)*(1-xHSstar/CHS) - d]]
    #  mutantS_eigenvalues = np.linalg.eigvals(mutantS_jacobian)
    #  realpartsS = [np.real(val) for val in mutantS_eigenvalues]
    #  print("mutantS eigvals", mutantS_eigenvalues)
    #  print("rounded realparts ", realpartsS)
     
     if R0S>1+10**(-7): # to exclude floating point errors
         trajectoryS.append(mutantS)
         print('mutantS accepted')
     else:
         trajectoryS.append(currentS)
         print('mutantS rejected')
     
def run_model(run_no):
     # Runs one simulation of the lattice model and saves the trajectory of traits of H and S at each mutation
     # time point. 
     # returns a list [run number, trait_trajectoryH, trait_trajectoryS]
     
     if verbose==True:
         print("started run number", run_no)
     trait_trajectoryH = [0.001]
     trait_trajectoryS = [0.001]

     for t in range(timesteps):
         print("\n NEW EVOL TIMESTEP")
         if t%20==0:
             print("now at evolution timestep ",t)
         omegaH = trait_trajectoryH[-1]
         omegaS = trait_trajectoryS[-1]
         print(t, omegaH, omegaS)
         # calculate fixed point of population dynamics
         try:
             xHstar, xSstar, xHSstar = GetPopDynSoln(omegaH, omegaS) 
             print('population sizes:', xHstar, xSstar, xHSstar)
         except:
             print("ode solver didn't converge. timesteps: ", len(trait_trajectoryH))
             print("trait value at this time. traitH = ", omegaH, "traitS = ", omegaS)
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
     
         # start the exponential clocks for host and symbiont populations
         clocks = [random.expovariate(xHstar), random.expovariate(xSstar)]
         if np.min(clocks) == clocks[0]:
             # the host clock went off first and so now we mutate the host trait
             MutateHost(omegaH,omegaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
             print('host mutated')
         if np.min(clocks) == clocks[1]:
             # the host clock went off first and so now we mutate the host trait
             MutateSymbiont(omegaH,omegaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
         elif clocks[0]==clocks[1]:
             # mutate host with probability 1/2, and symbiont with probability 1/2
             mutantpicker = random.uniform(0.0,1.0)
             if mutantpicker<0.5:
                 MutateHost(omegaH,omegaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
                 print('host mutated')
             elif 0.5<=mutantpicker<=1:
                 MutateSymbiont(omegaH,omegaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
                 print('symbiont mutated')

     return [run_no, trait_trajectoryH, trait_trajectoryS]

def record_result(result):
     # required for the "callback" arg in the multiprocessing.apply_async function to record the result given by each CPU 
     _, resultH, resultS = result
     global allrunsH
     global allrunsS
     global runlengths
     allrunsH.append(resultH)
     allrunsS.append(resultS)
     # Disclaimer: for the next line, this code arbtirarily chooses to store host trajectory under the assumption that both host and symbiont trajectories have the same length.
     runlengths.append(len(resultH))

if __name__ == '__main__':
     allrunsH = []
     allrunsS = []
     runlengths = []

     pool = mp.Pool(no_runs) # number of CPUs

     for run in range(1,no_runs+1):
         #  print('run number ',run, 'started out of ', no_runs)
         pool.apply_async(run_model, args=[run], callback=record_result)
        
     pool.close() # close the processes
     pool.join()  # finalise the processes by recording results
     
     # save data for later plotting
     for run in range(no_runs): 
         OmegaH_traj = np.around(allrunsH[run], 4)
         OmegaS_traj = np.around(allrunsS[run], 4)         
         data = pd.DataFrame(list(zip(OmegaH_traj, OmegaS_traj)), columns=['omegaH','omegaS'])
         data.to_csv("results/data/" + jobname + "/" + "obligacy-trajectory_" + jobname +"_run"+str(run)+".txt", index=False)
     
     # to make the final image smaller we work with a "zoomed out" array consisting only of every tenth element of the first array
     for r in range(no_runs):
         runH = allrunsH[r]
         runS = allrunsS[r]
         allrunsH[r] = runH[0::10]
         allrunsS[r] = runS[0::10]

     meanH = np.mean(allrunsH, axis=0)
     meanS = np.mean(allrunsS, axis=0)

     # to plot stochastic obligacy trajectories
     for run in range(no_runs):
         OmegaH_traj = np.around(allrunsH[run], 4)
         OmegaS_traj = np.around(allrunsS[run], 4)         
         plt.plot(OmegaH_traj, OmegaS_traj, '0.55', linewidth=0.75)  
             
     plt.plot(meanH, meanS, 'k', linewidth=2) # mean omegaH - omegaS trajectory

     plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--') # draw the line x=y
     plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5) # draw horizontal line y=1 
     plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5) # draw vertical line x=1 
     plt.xlim(0,1.1)
     plt.ylim(0,1.1)
     plt.gca().set_aspect('equal')
     plt.savefig("results/plots/" + jobname + "/" + "obligacy-trajectory_" + jobname  +".pdf",format='pdf')
     plt.show()
     plt.close()

     for trait_trajectoryH in allrunsH:
         plt.plot(range(len(trait_trajectoryH)), trait_trajectoryH, '0.55', linewidth=0.75)
    
     plt.plot(range(len(meanH)), meanH, 'k', linewidth=2)
     plt.plot(range(len(meanH)), [1.0]*len(meanH), 'k', linestyle='--')
     plt.ylim(0,1.1)
     plt.savefig("results/plots/" + jobname + "/" + "host-obligacy-trajectory_" + jobname +".pdf",format='pdf')
     plt.show()
     plt.close()

     for trait_trajectoryS in allrunsS:
         plt.plot(range(len(trait_trajectoryS)), trait_trajectoryS, '0.55', linewidth=0.75)
     
     plt.plot(range(len(meanS)), meanS, 'k', linewidth=2)
     plt.plot(range(len(meanS)), [1.0]*len(meanS), 'k', linestyle='--')
     plt.ylim(0,1.1)
     plt.savefig("results/plots/" + jobname + "/" + "symbiont-obligacy-trajectory_" + jobname  +".pdf",format='pdf')
     plt.show()



