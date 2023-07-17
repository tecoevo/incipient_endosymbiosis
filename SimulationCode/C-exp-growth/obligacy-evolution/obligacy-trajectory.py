from curses.ascii import RS
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint 
from matplotlib.lines import Line2D

verbose = True

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

allruns_OmegaH = []
allruns_OmegaS = []

# parameters (that ensure stability)

CH = 100.0
CS = 2*CH # we assume that the carrying capacity of the symbiont is twice that of the host
rH = 8.0
rS = 20.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
rHS = 10.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.
a = 0.1 
d = 50.0

# initialise lists that will hold values of the obligacies - these will be plotted against each other later to view the trajectory

timesteps = 1000
no_runs = 200

# we model evolution of the trait as a continuous-time Markov chain (CTMC).
# Mutations can arise, when the H-S-HS population is at dynamical equilibrium, in either the host or symbiont. 
# Which of these it will first arrive in is decided using the result that the waiting time between jumps of a CTMC 
# are exponentially distributed with rate equal to the transition rate between the two states. Here we only care about 
# which of the two mutants - host or symbiont - arises first and not about the exact time it takes, so we will use their 
# equilibrium population abundances as proxies. This is motivated by the fact that the coefficient of the selection  
# gradient in the canonical equation is composed of three terms - the mutation rate, the variance of the mutation  
# distribution, and the equilibrium population abundance. See the main text where we show that for this simple case, the 
# selection gradient is 1 for both. We assume that the mutational process for host and symbiont is similar, and so which 
# of the two obligacies grows faster is determined only by the equilibrium population abundance. 

mutation_std = 0.005
def MutateHost(currentH, currentS):
     # induce mutation in host obligacy and decide fate of the mutant. For derivation of
     # invasion criterion, see main text.
     OmegaS_trajectory.append(currentS)
     mutant_omegaH = np.clip(random.gauss(currentH,mutation_std), a_min=0, a_max=1)
     resident_objectiveH = rH*(1.0-currentH)*(1-d/(rHS*currentH*currentS))
     mutant_objectiveH = rH*(1.0-mutant_omegaH)*(1-d/(rHS*mutant_omegaH*currentS))
     if mutant_objectiveH > resident_objectiveH:
         OmegaH_trajectory.append(mutant_omegaH)
     else:
         OmegaH_trajectory.append(currentH)

def MutateSymbiont(currentH, currentS):
     OmegaH_trajectory.append(currentH)
     # induce mutation in symbiont obligacy and decide fate of the mutant. For derivation of
     # invasion criterion, see main text. 
     mutant_omegaS = np.clip(random.gauss(currentS,mutation_std), a_min=0, a_max=1)
     resident_objectiveS = rS*(1.0-currentS)*(1-d/(rHS*currentH*currentS))
     mutant_objectiveS = rS*(1.0-mutant_omegaS)*(1-d/(rHS*currentH*mutant_omegaS))
     if mutant_objectiveS > resident_objectiveS:
         OmegaS_trajectory.append(mutant_omegaS)
     else:
         OmegaS_trajectory.append(currentS)

Host=[]
Symb=[]

for run_no in range(no_runs): # different random trajectories
     host=0
     symbiont=0
     if verbose==True:
         print("started run number", run_no)
     OmegaH_trajectory = [0.001]
     OmegaS_trajectory = [0.001]
     for t in range(timesteps):
         current_omegaH = OmegaH_trajectory[-1]
         current_omegaS = OmegaS_trajectory[-1]
         fH = rH*(1-current_omegaH)
         fS = rS*(1-current_omegaS)
         fHS = rHS*current_omegaH*current_omegaS
         xHstar = (CH*fS*(fHS-d)*((d-fHS)*fH + a*CS*fHS))/((a**2)*CH*CS*(fHS**2)-fH*fS*((d-fHS)**2))
         xSstar = (CS*fH*(fHS-d)*((d-fHS)*fS +a*CH*fHS))/((a**2)*CH*CS*(fHS**2)-fH*fS*((d-fHS)**2))
    
         jacobian = [[fH*(1-2*xHstar/CH) - a*xSstar, -a*xHstar, d],
                     [-a*xSstar, fS*(1-2*xSstar/CS) - a*xHstar, d],
                     [a*xSstar, a*xHstar, fHS-d]]
        
         # for stability, we use the eigenvalues: the fixed point of the dynamical system giving
         # rise to the above Jacobian is stable iff all its eigenvalues have negative real part
        
         if d<= np.min([fHS*(1+a*np.sqrt(CH*CS/(fH*fS))), np.Inf]):
             if verbose==True:
                 print("Fixed point became unfeasible. Number of timesteps completed: ", len(OmegaH_trajectory))
             break
        
         eigenvalues = np.linalg.eigvals(jacobian)
         if np.max([np.real(val) for val in eigenvalues])>0:
             if verbose==True:
                 print("fixed point lost stability. Number of timesteps completed: ", len(OmegaH_trajectory))
             break
     
         # if any of the obligacies are 1, for example the host, then the host cannot live independently.
         # therefore, the population goes to zero abundance and no further mutants can arise.
         # strictly speaking, the below block isn't necessary because np.clip when generating the mutant trait
         # value should keep the trait at 1 no matter how many times Mutate is called. But removing this would 
         # lead to numerous useless Mutate calls and wasted time.
         if current_omegaH >= 1.0 and current_omegaS >= 1.0:
             break
         if current_omegaH >= 1.0:
             MutateSymbiont(current_omegaH,current_omegaS)
             continue
         if current_omegaS >= 1.0:
             MutateHost(current_omegaH,current_omegaS)
             continue
 
         # start the exponential clocks for host and symbiont populations
         clocks = [random.expovariate(xHstar), random.expovariate(xSstar)]
         if np.min(clocks) == clocks[0]:
             # the host clock went off first and so now we mutate the host trait
             MutateHost(current_omegaH,current_omegaS)
             host+=1
         if np.min(clocks) == clocks[1]:
             # the host clock went off first and so now we mutate the host trait
             MutateSymbiont(current_omegaH,current_omegaS)
             symbiont+=1
         elif clocks[0]==clocks[1]:
             # mutate host with probability 1/2, and symbiont with probability 1/2
             mutantpicker = random.uniform(0.0,1.0)
             if mutantpicker<0.5:
                 MutateHost(current_omegaH,current_omegaS)
                 host+=1
             elif 0.5<=mutantpicker<=1:
                 MutateSymbiont(current_omegaH,current_omegaS)
                 symbiont+=1
     Host.append(host/float(host+symbiont))
     Symb.append(symbiont/float(host+symbiont))
 
     allruns_OmegaH.append(OmegaH_trajectory)
     allruns_OmegaS.append(OmegaS_trajectory)

# now to plot the feasibility bound: we will keep it as an implicit equation of the Omegas and isolate
# the zero contours of (d - feasibility bound)

delta = 0.0005
xrange = np.arange(0.0, 1.0, delta)
yrange = np.arange(0.0, 1.0, delta)
X, Y = np.meshgrid(xrange,yrange)

F = d
G = rHS*X*Y*(1+a*np.sqrt((CH*CS)/(rH*rS*(1-X)*(1-Y))))

plt.contour(X, Y, (F - G), [0])

new_omegaH = []
new_omegaS = []
maxruntime = np.max([len(run) for run in allruns_OmegaH])
time_skip = 10 # plot trajectory points only every 10 timesteps

# to plot stochastic obligacy trajectories
for OmegaH_traj, OmegaS_traj in zip(allruns_OmegaH, allruns_OmegaS):
     OmegaH_traj = OmegaH_traj[0::time_skip]
     OmegaS_traj = OmegaS_traj[0::time_skip]
     plt.plot(OmegaH_traj, OmegaS_traj, '0.55', linewidth=0.75)

     newH, newS = HomogeniseLength(OmegaH_traj, OmegaS_traj, maxruntime)

     new_omegaH.append(newH)
     new_omegaS.append(newS)

mean_omegaH = np.nanmean(new_omegaH, axis=0)
mean_omegaS = np.nanmean(new_omegaS, axis=0)
plt.plot(mean_omegaH, mean_omegaS, 'k', linewidth=2)

plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.xlim(0,1.1)
plt.ylim(0,1.1)

custom_lines = [Line2D([0], [0], color='k', lw=2),
                Line2D([0], [0], color='k', linestyle='--', lw=2)]

plt.legend(custom_lines, ['Pre-image of feasibility bound', 'y=x'])
plt.gca().set_aspect('equal')
plt.savefig("obligacy-trajectory_rH="+str(rH) + "_rS=" + str(rS) +"_rHS=" + str(rHS)  +".pdf",format='pdf')
plt.show() 

# to plot f_i as a function of time: we need the arrays storing the obligacies for this

fHS_when_unstable = []

for i in range(no_runs):
     OmegaH_traj = allruns_OmegaH[i]
     OmegaH_traj = OmegaH_traj[0::time_skip]
     OmegaS_traj = allruns_OmegaS[i]
     OmegaS_traj = OmegaS_traj[0::time_skip]
     fH_array = rH*(1-np.array(OmegaH_traj))
     fS_array = rS*(1-np.array(OmegaS_traj))
     fHS_array = rHS*np.array(OmegaH_traj)*np.array(OmegaS_traj)
     fHS_when_unstable.append(fHS_array[-1])
     t = range(len(OmegaH_traj)) 
     plt.plot(t, fH_array, 'k', linewidth=0.75)
     plt.plot(t, fS_array, 'g', linewidth=0.75)
     plt.plot(t, fHS_array, 'b', linewidth=0.75)

# plot growth rates
maxt = range(round(np.max([len(Htraj) for Htraj in allruns_OmegaH])/time_skip))
plt.plot(maxt, [rH]*len(maxt), '--', color='k')
plt.plot(maxt, [rS]*len(maxt), '--', color='g')
plt.plot(maxt, [rHS]*len(maxt), '--', color='b')
plt.plot(maxt, [np.mean(fHS_when_unstable)]*len(maxt), linestyle='dotted', color='b')
plt.ylim([0,32])
plt.savefig("growthrates-trajectory_rH="+str(rH) + "_rS=" + str(rS) +"_rHS=" + str(rHS) +".pdf",format='pdf')
plt.show()
