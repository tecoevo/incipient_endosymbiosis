import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint 
from matplotlib.lines import Line2D

verbose = False

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

allruns_alphaH = []
allruns_alphaS = []

# parameters that ensure stability

CH = 100.0
CS = 2*CH # we assume that the carrying capacity of the symbiont is twice that of the host
fH = 8.0
fS = 20.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
rHS = 10.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.
a = 0.1 
d0 = 50.0

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
     # Briefly, mutants invade when they decrease the value of d/fHS
     alphaS_trajectory.append(currentS)
     mutant_alphaH = np.clip(random.gauss(currentH,mutation_std),a_min=0,a_max=1)
     resident_objectiveH = (rHS*currentH*currentS)/(d0*(1-currentH*currentS))
     mutant_objectiveH = (rHS*mutant_alphaH*currentS)/(d0*(1-mutant_alphaH*currentS))
     if mutant_objectiveH > resident_objectiveH:
         alphaH_trajectory.append(mutant_alphaH)
     else:
         alphaH_trajectory.append(currentH)

def MutateSymbiont(currentH, currentS):
     alphaH_trajectory.append(currentH)
     # induce mutation in symbiont obligacy and decide fate of the mutant. For derivation of
     # invasion criterion, see main text. 
     mutant_alphaS = np.clip(random.gauss(currentS,mutation_std),a_min=0,a_max=1)
     resident_objectiveS = (rHS*currentH*currentS)/(d0*(1-currentH*currentS))
     mutant_objectiveS = (rHS*currentH*mutant_alphaS)/(d0*(1-currentH*mutant_alphaS))
     if mutant_objectiveS > resident_objectiveS:
         alphaS_trajectory.append(mutant_alphaS)
     else:
         alphaS_trajectory.append(currentS)

for run_no in range(no_runs): # different random trajectories
     if verbose==True:
         print("started run number", run_no)
     alphaH_trajectory = [0.001]
     alphaS_trajectory = [0.001]
     for t in range(timesteps):
         current_alphaH = alphaH_trajectory[-1]
         current_alphaS = alphaS_trajectory[-1]
         d = d0*(1-current_alphaH*current_alphaS)
         fHS = rHS*current_alphaH*current_alphaS
         xHstar = (CH*fS*(fHS-d)*((d-fHS)*fH + a*CS*fHS))/((a**2)*CH*CS*(fHS**2)-fH*fS*((d-fHS)**2))
         xSstar = (CS*fH*(fHS-d)*((d-fHS)*fS +a*CH*fHS))/((a**2)*CH*CS*(fHS**2)-fH*fS*((d-fHS)**2))
    
         jacobian = [[fH*(1-2*xHstar/CH) - a*xSstar, -a*xHstar, d],
                     [-a*xSstar, fS*(1-2*xSstar/CS) - a*xHstar, d],
                     [a*xSstar, a*xHstar, fHS-d]]
        
         # for stability, we use the eigenvalues: the fixed point of the dynamical system giving
         # rise to the above Jacobian is stable iff all its eigenvalues have negative real part
        
         if d<= np.min([fHS*(1+a*np.sqrt(CH*CS/(fH*fS))), np.Inf]):
             if verbose==True:
                 print("Fixed point became unfeasible. Number of timesteps completed: ", len(alphaH_trajectory))
             break
        
         eigenvalues = np.linalg.eigvals(jacobian)
         if np.max([np.real(val) for val in eigenvalues])>0:
             if verbose==True:
                 print("fixed point lost stability. Number of timesteps completed: ", len(alphaH_trajectory))
             break
     
         # if any of the oligacies are 1, for example the host, then the host cannot live independently.
         # therefore, the population goes to zero abundance and no further mutants can arise.
         # strictly speaking, the below block isn't necessary because np.clip when generating the mutant trait
         # value should keep the trait at 1 no matter how many times Mutate is called. But removing this would 
         # lead to numerous useless Mutate calls and wasted time.

         if current_alphaH >= 1.0 and current_alphaS >= 1.0:
             break
         if current_alphaH >= 1.0:
             MutateSymbiont(current_alphaH,current_alphaS)
             continue
         if current_alphaS >= 1.0:
             MutateHost(current_alphaH,current_alphaS)
             continue
 
         # start the exponential clocks for host and symbiont populations
         clocks = [random.expovariate(xHstar), random.expovariate(xSstar)]
         if np.min(clocks) == clocks[0]:
             # the host clock went off first and so now we mutate the host trait
             MutateHost(current_alphaH,current_alphaS)
         if np.min(clocks) == clocks[1]:
             # the host clock went off first and so now we mutate the host trait
             MutateSymbiont(current_alphaH,current_alphaS)
         elif clocks[0]==clocks[1]:
             # mutate host with probability 1/2, and symbiont with probability 1/2
             mutantpicker = random.uniform(0.0,1.0)
             if mutantpicker<0.5:
                 MutateHost(current_alphaH,current_alphaS)
             elif 0.5<=mutantpicker<=1:
                 MutateSymbiont(current_alphaH,current_alphaS)
     
     allruns_alphaH.append(alphaH_trajectory)
     allruns_alphaS.append(alphaS_trajectory)

# now we numerically integrate the canonical equation to see if these paths are reproduced. 

# def f(y, t):
#      # we shall have "dummy" variables just for the purposes of this function so that it doesn't 
#      # interfere with the rest of the script. 
#      dummy_omegaH = y[0]
#      dummy_omegaS = y[1]
#      dummy_fH = fH*(1-dummy_omegaH)
#      dummy_fS = fS*(1-dummy_omegaS)
#      dummy_fHS = rHS*dummy_omegaH*dummy_omegaS
#      dummy_xHstar = (CH*dummy_fS*(dummy_fHS-d)*((d-dummy_fHS)*dummy_fH + a*CS*dummy_fHS))/((a**2)*CH*CS*(dummy_fHS**2)-dummy_fH*dummy_fS*((d-dummy_fHS)**2))
#      dummy_xSstar = (CS*dummy_fH*(dummy_fHS-d)*((d-dummy_fHS)*dummy_fS + a*CH*dummy_fHS))/((a**2)*CH*CS*(dummy_fHS**2)-dummy_fH*dummy_fS*((d-dummy_fHS)**2))
#      f0 = dummy_xHstar
#      f1 = dummy_xSstar
#      return [f0, f1]

# Omega_0 = [0.0, 0.0] # vector of initial obligacies
# t  = np.linspace(0, 0.01, 100) # time grid for numerical integration 

# soln_CE = odeint(f, Omega_0, t)
# num_omegaH = np.array(soln_CE[:, 0])
# num_omegaS = np.array(soln_CE[:, 1])

# # playing around with the numerical solution
# num_fH = rH*(1-num_omegaH)
# num_fS = rS*(1-num_omegaS)
# num_fHS = rHS*num_omegaH*num_omegaS
# flow_H = (CH*num_fS*(num_fHS-d)*((d-num_fHS)*num_fH + a*CS*num_fHS))/((a**2)*CH*CS*(num_fHS**2)-num_fH*num_fS*((d-num_fHS)**2))
# flow_S = (CS*num_fH*(num_fHS-d)*((d-num_fHS)*num_fS + a*CH*num_fHS))/((a**2)*CH*CS*(num_fHS**2)-num_fH*num_fS*((d-num_fHS)**2))
# plt.plot(t, flow_S)
# plt.show()

# # plotting mean path of CE
# plt.plot(num_omegaH, num_omegaS, 'k', linewidth=2, label='Mean path (solution of CE)')

# now to plot the feasibility bound: we will keep it as an implicit equation of the Omegas and isolate
# the zero contours of (d - feasibility bound)

delta = 0.0005
xrange = np.arange(0.0, 1.0, delta)
yrange = np.arange(0.0, 1.0, delta)
X, Y = np.meshgrid(xrange,yrange)

F = d0*(1-X*Y)
G = rHS*X*Y*(1+a*np.sqrt((CH*CS)/(fH*fS)))

plt.contour(X, Y, (F - G), [0])

new_alphaH = []
new_alphaS = []
maxruntime = np.max([len(run) for run in allruns_alphaH])
time_skip = 10 # plot trajectory points only every 10 timesteps

# to plot stochastic obligacy trajectories
for alphaH_traj, alphaS_traj in zip(allruns_alphaH, allruns_alphaS):
     alphaH_traj = alphaH_traj[0::time_skip]
     alphaS_traj = alphaS_traj[0::time_skip]

     plt.plot(alphaH_traj, alphaS_traj, '0.55', linewidth=0.75)

     newH, newS = HomogeniseLength(alphaH_traj, alphaS_traj, maxruntime)

     new_alphaH.append(newH)
     new_alphaS.append(newS)

mean_alphaH = np.nanmean(new_alphaH, axis=0)
mean_alphaS = np.nanmean(new_alphaS, axis=0)
plt.plot(mean_alphaH, mean_alphaS, 'k', linewidth=2)

plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.xlim(0,1.1)
plt.ylim(0,1.1)

custom_lines = [Line2D([0], [0], color='k', lw=2),
                Line2D([0], [0], color='k', linestyle='--', lw=2)]

plt.legend(custom_lines, ['Pre-image of feasibility bound', 'y=x'])
plt.gca().set_aspect('equal')
plt.savefig("stickiness-trajectory_d0=" + str(d0) + "_rHS=" + str(rHS) + ".pdf",format='pdf')
plt.show()

# to plot f_i as a function of time: we need the arrays storing the obligacies for this

fHS_when_unstable = []

for i in range(no_runs):
     alphaH_traj = allruns_alphaH[i]
     alphaH_traj = alphaH_traj[0::time_skip]
     alphaS_traj = allruns_alphaS[i]
     alphaS_traj = alphaS_traj[0::time_skip]
     d_array = d0*(1-np.array(alphaH_traj)*np.array(alphaS_traj))
     fHS_array = rHS*np.array(alphaH_traj)*np.array(alphaS_traj)
     fHS_when_unstable.append(fHS_array[-1])
     t = range(len(alphaH_traj))
     if i==0:
         plt.plot(t, d_array, 'r', linewidth=0.75, label='d, individual runs')
         plt.plot(t, fHS_array, 'b', linewidth=0.75, label='fHS, individual runs')
     else:
         plt.plot(t, d_array, 'r', linewidth=0.75)
         plt.plot(t, fHS_array, 'b', linewidth=0.75)
     
# print(np.mean(fHS_when_unstable), np.std(fHS_when_unstable))

# plot growth rates
maxt = range(round(np.max([len(Htraj) for Htraj in allruns_alphaH])/time_skip))
plt.plot(maxt, [d0]*len(maxt), '--', color='r')
plt.plot(maxt, [rHS]*len(maxt), '--', color='b')
plt.plot(maxt, [np.mean(fHS_when_unstable)]*len(maxt), linestyle='dotted', color='b')
plt.ylim([0,60])
plt.savefig("growthrates-trajectory_d0=" + str(d0) + "_rHS=" + str(rHS) + ".pdf",format='pdf')
plt.show()
