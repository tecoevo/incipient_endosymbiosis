import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint 
from matplotlib.lines import Line2D
import pandas as pd 
from scipy.interpolate import CubicSpline

verbose = True

def fH(omegaH):
     return rH*(1.-omegaH)

def fS(omegaS):
     return rS*(1.-omegaS)

def fHS(omegaH, omegaS, alphaH, alphaS):
     return rHS*(omegaH*omegaS)*(alphaH*alphaS)

def d(alphaH, alphaS):
     return d0*(1.-(alphaH*alphaS))

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

def Interpolate(oldx,oldy, step):
     # given a pair of lists of equal length with entries from [0,1], output a list of pair of lists s.t.
     # the new pair of lists has a user-determined length (determined by the `newlen` argument).
     # new list is made by artificially extending the old list such that there are new copies of the last
     # elements of the old lists - as many as necessary to get them up to the required length.
     
     newx = np.arange(0,1,step)
     newx = newx + float(step)/10
     newy = np.zeros(len(newx))

     for i in range(len(newx)):
         # find all indices at which newx[i] occurs in oldx
         indices = [j for j, value in enumerate(oldx) if value == newx[i]]
         if len(indices)>0: # if there is at least one occurrence
             newy[i] = np.nanmean([oldx[j] for j in indices]) # then take the corresponding y value
         if len(indices)==0: # if there are no occurrences
             # then find the two points of oldx between which newx[i] lies (possibly >1).
             # for every x-interval that newx[i] falls into, we calculate a corresponding y-value from the
             # interpolating line and put it into the following list
             potential_newy_list = [] 
             for k in range(len(oldx)-1):
                 if oldx[k] <= newx[i] < oldx[k+1]:
                     # linearly interpolate between the 2d points (oldx[k],oldy[k]) and (oldx[k+1], oldy[k+1])
                     x1 = float(oldx[k])
                     y1 = float(oldy[k])
                     x2 = float(oldx[k+1])
                     y2 = float(oldy[k+1])
                     # now using the equation of the interpolating line, calculate y-value at newx[i]
                     pot_newy = y1*(x2 - newx[i])/(x2-x1) + y2*(newx[i]-x1)/(x2-x1)
                     potential_newy_list.append(pot_newy)             
             
             if len(potential_newy_list)==0: 
                 # then there were no y-values for this newx value. 
                 # this means the trajectory does not "go this far", and our homogenised list should also not. 
                 # so we attach a nan to all such x values
                 newy[i] = np.nan
             else: # now average over all potential y values, of which there is at least one
                 newy[i] = np.nanmean(potential_newy_list)
     
     return (np.array(newx), np.array(newy))

# to store evolutionary trajectories

allrunsH = []
allrunsS = []

# parameters that ensure stability

CH = 100.0
CS = 2*CH # we assume that the carrying capacity of the symbiont is twice that of the host
rH = 8.0
rS = 20.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
rHS = 10.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.
a = 0.1
d0 = 50.0

# initialise lists that will hold values of the obligacies - these will be plotted against each other later to view the trajectory

timesteps = 1000
no_runs = 200

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

Host = []
Symb = []

def Objective(traitH, traitS, id):
     omegaH = traitH[0]
     omegaS = traitS[0]
     alphaH = traitH[1]
     alphaS = traitS[1]
     if id=='H':
         return (fH(omegaH)/a)*(1-(d(alphaH, alphaS))/fHS(omegaH, omegaS, alphaH, alphaS))
     if id=='S':
         return (fS(omegaS)/a)*(1-(d(alphaH, alphaS))/fHS(omegaH, omegaS, alphaH, alphaS))
 
def MutateHost(currentH, currentS):
     # induce mutation in host obligacy and decide fate of the mutant. For derivation of
     # invasion criterion, see main text.
     # Briefly, mutants invade when they decrease the value of d/fHS
     trait_trajectoryS.append(currentS)
     mutantH = np.clip(np.random.multivariate_normal(currentH, [[mutation_std, 0], [0, mutation_std]])
     ,a_min=0,a_max=1)
     resident_objectiveH = Objective(currentH, currentS, 'H') 
     mutant_objectiveH = Objective(mutantH, currentS, 'H')
     if mutant_objectiveH > resident_objectiveH:
         trait_trajectoryH.append(mutantH)
     else:
         trait_trajectoryH.append(currentH)

def MutateSymbiont(currentH, currentS):
     trait_trajectoryH.append(currentH)
     # induce mutation in symbiont obligacy and decide fate of the mutant. For derivation of
     # invasion criterion, see main text. 
     mutantS = np.clip(np.random.multivariate_normal(currentS, [[mutation_std, 0], [0, mutation_std]])
     ,a_min=0,a_max=1)
     resident_objectiveS = Objective(currentH, currentS, 'S') 
     mutant_objectiveS = Objective(currentH, mutantS, 'S') 
     if mutant_objectiveS > resident_objectiveS:
         trait_trajectoryS.append(mutantS)
     else:
         trait_trajectoryS.append(currentS)

for run_no in range(no_runs): # different random trajectories
     host=0
     symbiont=0
     if verbose==True:
         print("started run number", run_no)
     trait_trajectoryH = [[0.001, 0.001]]
     trait_trajectoryS = [[0.001, 0.001]]
     for t in range(timesteps):
         currentH = trait_trajectoryH[-1]
         currentS = trait_trajectoryS[-1]
         omegaH = currentH[0]
         omegaS = currentS[0]
         alphaH = currentH[1]
         alphaS = currentS[1]

         fH_here = fH(omegaH)
         fS_here = fS(omegaS)
         fHS_here = fHS(omegaH, omegaS, alphaH, alphaS)
         d_here = d(alphaH, alphaS) 

         xHstar = (CH*fS_here)*(fHS_here-d_here)*((d_here-fHS_here)*fH_here + a*CS*fHS_here)/((a**2)*CH*CS*(fHS_here**2)-fH_here*fS_here*((d_here-fHS_here)**2))
         xSstar = (CS*fH_here*(fHS_here-d_here)*((d_here-fHS_here)*fS_here + a*CH*fHS_here))/((a**2)*CH*CS*(fHS_here**2)-fH_here*fS_here*((d_here-fHS_here)**2))
    
         jacobian = [[fH_here*(1-2*xHstar/CH) - a*xSstar, -a*xHstar, d_here],
                     [-a*xSstar, fS_here*(1-2*xSstar/CS) - a*xHstar, d_here],
                     [a*xSstar, a*xHstar, fHS_here-d_here]]
        
         # for stability, we use the eigenvalues: the fixed point of the dynamical system giving
         # rise to the above Jacobian is stable iff all its eigenvalues have negative real part
        
         if d_here<= np.min([fHS_here*(1+a*np.sqrt(CH*CS/(fH_here*fS_here))), np.Inf]):
             if verbose==True:
                 print("Fixed point became unfeasible. Number of timesteps completed: ", len(trait_trajectoryH))
             break
         
         eigenvalues = np.linalg.eigvals(jacobian)
         if np.max([np.real(val) for val in eigenvalues])>0:
             if verbose==True:
                 print("fixed point lost stability. Number of timesteps completed: ", len(trait_trajectoryH))
             break
     
         # if any of the oligacies are 1, for example the host, then the host cannot live independently.
         # therefore, the population goes to zero abundance and no further mutants can arise.
         # strictly speaking, the below block isn't necessary because np.clip when generating the mutant trait
         # value should keep the trait at 1 no matter how many times Mutate is called. But removing this would 
         # lead to numerous useless Mutate calls and wasted time.

         if any(np.array(currentH) >= 1.0) and any(np.array(currentS) >= 1):
             break
         if any(np.array(currentH) >= 1.0) >= 1.0:
             MutateSymbiont(currentH,currentS)
             continue
         if any(np.array(currentS) >= 1) >= 1.0:
             MutateHost(currentH,currentS)
             continue
 
         # start the exponential clocks for host and symbiont populations
         clocks = [random.expovariate(xHstar), random.expovariate(xSstar)]
         if np.min(clocks) == clocks[0]:
             # the host clock went off first and so now we mutate the host trait
             host+=1
             MutateHost(currentH,currentS)
         if np.min(clocks) == clocks[1]:
             # the host clock went off first and so now we mutate the host trait
             symbiont+=1
             MutateSymbiont(currentH,currentS)
         elif clocks[0]==clocks[1]:
             # mutate host with probability 1/2, and symbiont with probability 1/2
             mutantpicker = random.uniform(0.0,1.0)
             if mutantpicker<0.5:
                 host+=1
                 MutateHost(currentH,currentS)
             elif 0.5<=mutantpicker<=1:
                 symbiont+=1
                 MutateSymbiont(currentH,currentS)
     Host.append(host/float(host+symbiont))
     Symb.append(symbiont/float(host+symbiont))
     allrunsH.append(trait_trajectoryH)
     allrunsS.append(trait_trajectoryS)

########################## plotting ###############################

# note: we cannot plot the feasibility bound now because it is a function now of 
# 4 parameters - omega_i, alpha_i for i=H,S. 

# to plot stochastic evolutionary trajectories 

# first the obligacies - omegaH vs. omegaS

new_omegaH = []
new_omegaS = []
maxruntime = np.max([[len(run) for run in allrunsH]])
timeskip = 1

for run in range(no_runs):
     traitH = np.array(allrunsH[run])
     traitS = np.array(allrunsS[run])

     omegaH_traj = np.around(traitH[:,0], 4)
     omegaS_traj= np.around(traitS[:,0], 4)
     omegaH_traj = omegaH_traj[0::timeskip]
     omegaS_traj = omegaS_traj[0::timeskip]
     plt.plot(omegaH_traj, omegaS_traj, '0.55', linewidth=0.85)

     newH, newS = HomogeniseLength(omegaH_traj, omegaS_traj, maxruntime)

     new_omegaH.append(newH)
     new_omegaS.append(newS)

mean_omegaH = np.nanmean(new_omegaH, axis=0)
mean_omegaS = np.nanmean(new_omegaS, axis=0)
plt.plot(mean_omegaH, mean_omegaS, 'k', linewidth=2)

plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.savefig("coevolution_omegas_omegaH-x_omegaS-y.pdf", format='pdf')
plt.close()

# now the adhesions - alphaH vs. alphaS

new_alphaH = []
new_alphaS = []

for run in range(no_runs):
     traitH = np.array(allrunsH[run])
     traitS = np.array(allrunsS[run])

     alphaH_traj = np.around(traitH[:,1], 4)
     alphaS_traj = np.around(traitS[:,1], 4)
     alphaH_traj = alphaH_traj[0::timeskip]
     alphaS_traj = alphaS_traj[0::timeskip]
     plt.plot(alphaH_traj, alphaS_traj, '0.55', linewidth=0.85)

     newH, newS = HomogeniseLength(alphaH_traj, alphaS_traj, maxruntime)

     new_alphaH.append(newH)
     new_alphaS.append(newS)

mean_alphaH = np.nanmean(new_alphaH, axis=0)
mean_alphaS = np.nanmean(new_alphaS, axis=0)
plt.plot(mean_alphaH, mean_alphaS, 'k', linewidth=2)

plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.savefig("coevolution_alphas_alphaH-x_alphaS-y.pdf", format='pdf')
plt.close()

# now the host traits - omegaH vs. alphaH
new_omegaH = []
new_alphaH = []

for run in range(no_runs):
     traitH = np.array(allrunsH[run])
     omegaH_traj = np.around(traitH[:,0], 4)
     alphaH_traj = np.around(traitH[:,1], 4)
     
     omegaH_traj = omegaH_traj[0::timeskip]
     alphaH_traj = alphaH_traj[0::timeskip]

     plt.plot(omegaH_traj, alphaH_traj, '0.55', linewidth=0.85)

     newomegaH, newalphaH = HomogeniseLength(omegaH_traj, alphaH_traj, maxruntime)

     new_omegaH.append(newomegaH)
     new_alphaH.append(newalphaH)

mean_omegaH = np.nanmean(new_omegaH, axis=0)
mean_alphaH = np.nanmean(new_alphaH, axis=0)
plt.plot(mean_omegaH, mean_alphaH, 'k', linewidth=2)

plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.savefig("coevolution_hosttraits_omegaH-x_alphaH-y.pdf", format='pdf')
plt.close()

# the symbiont traits - omegaS vs. alphaS
new_omegaS = []
new_alphaS = []

for run in range(no_runs):
     traitS = np.array(allrunsS[run])
     omegaS_traj= np.around(traitS[:,0], 4)
     alphaS_traj = np.around(traitS[:,1], 4)

     omegaS_traj = omegaS_traj[0::timeskip]
     alphaS_traj = alphaS_traj[0::timeskip]

     plt.plot(omegaS_traj, alphaS_traj, '0.55', linewidth=0.85)
     newomegaS, newalphaS = HomogeniseLength(omegaS_traj, alphaS_traj, maxruntime)

     new_omegaS.append(newomegaS)
     new_alphaS.append(newalphaS)

mean_omegaS = np.nanmean(new_omegaS, axis=0)
mean_alphaS = np.nanmean(new_alphaS, axis=0)
plt.plot(mean_omegaS, mean_alphaS, 'k', linewidth=2)

plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.savefig("coevolution_symbionttraits_omegaS-x_alphaS-y.pdf", format='pdf')
plt.close()

# coevolution of mutual dependence and reproductive cohesion 

cohesions = []
dependences = []

# dependence-cohesion
for run in range(no_runs):
     traitH = np.array(allrunsH[run])
     traitS = np.array(allrunsS[run])

     omegaH_traj = np.around(traitH[:,0], 4)
     omegaS_traj= np.around(traitS[:,0], 4)
     omegaH_traj = omegaH_traj[0::timeskip]
     omegaS_traj = omegaS_traj[0::timeskip]
     dependence = (omegaH_traj*omegaS_traj)

     alphaH_traj = np.around(traitH[:,1], 4)
     alphaS_traj = np.around(traitS[:,1], 4)
     alphaH_traj = alphaH_traj[0::timeskip]
     alphaS_traj = alphaS_traj[0::timeskip]
     cohesion = (alphaH_traj*alphaS_traj)
     plt.plot(dependence, cohesion, '0.55', linewidth=0.85)

     new_dependence, new_cohesion = HomogeniseLength(dependence, cohesion, maxruntime)

     dependences.append(new_dependence)
     cohesions.append(new_cohesion)

mean_dependence = np.nanmean(dependences, axis=0)
mean_cohesion = np.nanmean(cohesions, axis=0)
print(len(mean_cohesion), len(mean_dependence))

plt.plot(mean_dependence, mean_cohesion, 'k', linewidth=2)
plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.savefig("coevolution_dependence-x_cohesion-y_d0=" + str(d0) + "_rHS=" + str(rHS) + ".pdf",format='pdf')
plt.close()
