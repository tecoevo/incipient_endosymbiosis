import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.lines import Line2D
import multiprocessing as mp
import pandas as pd 
import os

verbose = True

# to store evolutionary trajectories
# parameters that ensure stability

CH = 100.0 
CS = 2*CH # we assume that the carrying capacity of the symbiont is twice that of the host
CHS = 250.0

rH = 8.0
rS = 20.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
rHS = 10.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.

a = 0.1 
d = 50.0

dH = 0.0
dS = dH

bH = 25.0
bS = bH

init_trait_value = 0.001
init_trait_valueH = init_trait_value
init_trait_valueS = init_trait_value

# initialise lists that will hold values of the obligacies - these will be plotted against each other later to view the trajectory

timesteps = 5000
no_runs = 20

# make sure all directories necessary to store results exist
jobname = 'd=' + str(d) + '_dH=' + str(dH) + '_dS=' + str(dS) + '_bH=' + str(bH) + '_bS=' + str(bS) + '_inittrait=' + str(init_trait_value) # if there is further subdivision within the folder 'data'. if not, leave empty

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

mutation_std = 0.005

def fH(omegaH):
     return rH*(1.-omegaH)

def fS(omegaS):
     return rS*(1.-omegaS)

def fHS(omegaH, omegaS):
     return rHS*omegaH*omegaS

def GetPopDynSoln(omegaH, omegaS):
     steps = 100000000
     
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
         xH += dt*(fH(omegaH)*xH*(1-xH/CH) - a*xH*xS + d*xHS + dS*xHS + bH*xHS)
         xS += dt*(fS(omegaS)*xS*(1-xS/CS) - a*xH*xS + d*xHS + dH*xHS + bS*xHS)
         xHS += dt*(fHS(omegaH, omegaS)*xHS*(1-xHS/CHS) + a*xH*xS - d*xHS - dS*xHS - dH*xHS)

         if time%10000 == 0: # check every so often if the routine has converged
             if xH - np.mean(xHarray[-1:-10:-1])<0.0001 and \
             xS - np.mean(xSarray[-1:-10:-1])<0.0001 and \
             xHS - np.mean(xHSarray[-1:-10:-1])<0.0001:
                 return [xH, xS, xHS]

         elif time==steps:
             print("Maximum time exceeded. Perhaps the fixed point doesn't exist/isn't stable. Here is the last point of the ")
             return [xH, xS, xHS]

def MutateHost(currentH, currentS, xHstar, xSstar, xHSstar, trajectoryH, trajectoryS):
     # induce mutation in host obligacy and decide fate of the mutant. For derivation of
     # invasion criterion, see main text.
     # Briefly, mutants invade when they decrease the value of d/fHS
     trajectoryS.append(currentS)
     mutantH = np.clip(random.gauss(currentH,mutation_std), a_min=0, a_max=1)
    #  print('mutantH value: ', mutantH)

    #  host_inflow_dueto_encounterrateH = (d(mutantH, currentS)*a*xSstar)/(d(mutantH, currentS) - fHS(mutantH, currentS) + (fHS(mutantH, currentS)*xHSstar)/(CHS))
    #  host_inflow_dueto_independentreproduction = fH
    #  host_outflow = a*xSstar + (fH*xHstar)/(CH)
    #  R0H = (host_inflow_dueto_independentreproduction + host_inflow_dueto_encounterrateH)/host_outflow
    #  print("R0H = ", R0H)
    #  if R0H>1+10**(-7):
    #      trajectoryH.append(mutantH)
    #      print('mutantH accepted')
    #  else:
    #      trajectoryH.append(currentH)
    #      print('mutantH rejected')

     mutantH_jacobian = [[fH(mutantH)*(1-xHstar/CH) - a*xSstar, d+dS+bH], 
     [a*xSstar, fHS(mutantH, currentS)*(1-xHSstar/CHS) - d-dS-dH]]
     mutantH_eigenvalues = np.linalg.eigvals(mutantH_jacobian)
     realpartsH = [np.real(val) for val in mutantH_eigenvalues]
    #  print('real parts H: ', realpartsH)
     
     if np.max(realpartsH)>1e-07:
         trajectoryH.append(mutantH)
        #  print('mutantH accepted')
     else:
         trajectoryH.append(currentH)
        #  print('mutantH rejected')

def MutateSymbiont(currentH, currentS, xHstar, xSstar, xHSstar, trajectoryH, trajectoryS):
     # induce mutation in host obligacy and decide fate of the mutant. For derivation of
     # invasion criterion, see main text.
     # Briefly, mutants invade when they decrease the value of d/fHS
     trajectoryH.append(currentH)
     mutantS = np.clip(random.gauss(currentS,mutation_std), a_min=0, a_max=1)
    #  print('mutantS value: ', mutantS)

    #  symb_inflow_dueto_encounterrateH = (d(currentH, mutantS)*a*xHstar)/(d(currentH, mutantS) - fHS(currentH, mutantS) + (fHS(currentH, mutantS)*xHSstar)/(CHS))
    #  symb_inflow_dueto_independentreproduction = fS
    #  symb_outflow = a*xHstar + (fS*xSstar)/(CS)
    #  R0S = (symb_inflow_dueto_independentreproduction + symb_inflow_dueto_encounterrateH)/symb_outflow
    #  print("R0S = ", R0S)
    #  if R0S>1+10**(-7):
    #      trajectoryS.append(mutantS)
    #      print('mutantS accepted')
    #  else:
    #      trajectoryS.append(currentS)
    #      print('mutantS rejected')
     
     mutantS_jacobian = [[fS(mutantS)*(1-xSstar/CS) - a*xHstar, d+dH+bS], 
     [a*xHstar, fHS(currentH, mutantS)*(1-xHSstar/CHS) - d - dS - dH]]
     mutantS_eigenvalues = np.linalg.eigvals(mutantS_jacobian)
     realpartsS = [np.real(val) for val in mutantS_eigenvalues]
    #  print('real parts S: ', realpartsS)
     
     if np.max(realpartsS)>1e-07:
         trajectoryS.append(mutantS)
        #  print('mutantS accepted')
     else:
         trajectoryS.append(currentS)
        #  print('mutantS rejected')
     
def run_model(run_no):
     # Runs one simulation of the lattice model and saves the trajectory of traits of H and S at each mutation
     # time point. 
     # returns a list [run number, trait_trajectoryH, trait_trajectoryS]
     
     if verbose==True:
         print("started run number", run_no)

     trait_trajectoryH = [init_trait_valueH]
     trait_trajectoryS = [init_trait_valueS]

     for t in range(timesteps):
        #  print("\n NEW EVOL TIMESTEP")
         if t%20==0:
             print("now at evolution timestep ",t)
         omegaH = trait_trajectoryH[-1]
         omegaS = trait_trajectoryS[-1]
        #  print(t, omegaH, omegaS)
         # calculate fixed point of population dynamics
         try:
             xHstar, xSstar, xHSstar = GetPopDynSoln(omegaH, omegaS)
            #  print('population sizes:', xHstar, xSstar, xHSstar)
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
             MutateHost(omegaH, omegaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
            #  print('host mutated')
         if np.min(clocks) == clocks[1]:
             # the host clock went off first and so now we mutate the host trait
             MutateSymbiont(omegaH, omegaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
            #  print('symbiont mutated')
         elif clocks[0]==clocks[1]:
             # mutate host with probability 1/2, and symbiont with probability 1/2
             mutantpicker = random.uniform(0.0,1.0)
             if mutantpicker<0.5:
                 MutateHost(omegaH, omegaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)
             elif 0.5<=mutantpicker<=1:
                 MutateSymbiont(omegaH, omegaS, xHstar, xSstar, xHSstar, trait_trajectoryH, trait_trajectoryS)

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

     # save data for later plotting
     for run in range(no_runs): 
         alphaH_traj = np.around(allrunsH[run], 4)
         alphaS_traj = np.around(allrunsS[run], 4)         
         data = pd.DataFrame(list(zip(alphaH_traj, alphaS_traj)), columns=['alphaH','alphaS'])
         data.to_csv("results/data/" + jobname + "/" + "obligacy-trajectory_" + jobname +"_run"+str(run)+".txt", index=False)
     
     # to make the final image smaller we work with a "zoomed out" array consisting only of every tenth element of the first array
     for r in range(no_runs):
         runH = allrunsH[r]
         runS = allrunsS[r]
         allrunsH[r] = runH[0::10]
         allrunsS[r] = runS[0::10]

     meanH = np.mean(allrunsH, axis=0)
     meanS = np.mean(allrunsS, axis=0)

     # plot obligacy trajectories
     for run in range(no_runs):
         alphaH_traj = np.around(allrunsH[run], 4)
         alphaS_traj = np.around(allrunsS[run], 4)         
         plt.plot(alphaH_traj, alphaS_traj, '0.55', linewidth=0.75)  
             
     plt.plot(meanH, meanS, 'k', linewidth=2) # mean alphaH - alphaS trajectory

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


