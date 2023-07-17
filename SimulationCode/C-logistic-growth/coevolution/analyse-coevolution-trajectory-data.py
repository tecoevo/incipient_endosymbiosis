import numpy as np
import matplotlib.pyplot as plt
import os

rH=8.0
rS=20.0
rHS=10.0
d0=50.0

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

srcdir = './results/data/ODEstep_e-04/' # in which directory are the trajectory files that you want to analyse located?
                                        # make sure that this name starts with ./ and has / at the end

time_skip=1

allrunsOmegaH = []
allrunsalphaH = []
allrunsOmegaS = []
allrunsalphaS = []
runlengths = []

# record all trajectories
for fil in os.listdir(srcdir):
     filename = srcdir + fil

     omegaH_traj = np.loadtxt(filename, dtype='float', delimiter=',', skiprows=1, usecols=0)
     omegaH_traj = omegaH_traj[0::time_skip]
     allrunsOmegaH.append(omegaH_traj)

     alphaH_traj = np.loadtxt(filename, dtype='float', delimiter=',', skiprows=1, usecols=1)
     alphaH_traj = alphaH_traj[0::time_skip]
     allrunsalphaH.append(alphaH_traj)

     omegaS_traj = np.loadtxt(filename, dtype='float', delimiter=',', skiprows=1, usecols=2)
     omegaS_traj = omegaS_traj[0::time_skip]
     allrunsOmegaS.append(omegaS_traj)

     alphaS_traj = np.loadtxt(filename, dtype='float', delimiter=',', skiprows=1, usecols=3)
     alphaS_traj = alphaS_traj[0::time_skip]
     allrunsalphaS.append(alphaS_traj)

     runlengths.append(len(omegaH_traj))

maxruntime = np.max(runlengths)

##### plotting #####

# host traits 
newrunsOmegaH = []
newrunsalphaH = []

for omegaHrun, alphaHrun in zip(allrunsOmegaH, allrunsalphaH):
      newomega, newalpha = HomogeniseLength(omegaHrun, alphaHrun, maxruntime)
      plt.plot(omegaHrun, alphaHrun, '0.55', linewidth=0.65)

      newrunsOmegaH.append(newomega)
      newrunsalphaH.append(newalpha)

mean_omegaH = np.nanmean(newrunsOmegaH, axis=0)
mean_alphaH = np.nanmean(newrunsalphaH, axis=0)
plt.plot(mean_omegaH, mean_alphaH, 'k', linewidth=2)

plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.gca().set_aspect('equal')
# plt.savefig("coevolution_dependence-x_cohesion-y_d0=" + str(d0) + "_rHS=" + str(rHS) + ".pdf",format='pdf')
plt.show()

# # symbiont traits
newrunsOmegaS = []
newrunsalphaS = []
 
for omegaSrun, alphaSrun in zip(allrunsOmegaS, allrunsalphaS):
      newomega, newalpha = HomogeniseLength(omegaSrun, alphaSrun, maxruntime)
      plt.plot(omegaSrun, alphaSrun, '0.55', linewidth=0.65)

      newrunsOmegaS.append(newomega)
      newrunsalphaS.append(newalpha)

mean_omegaS = np.nanmean(newrunsOmegaS, axis=0)
mean_alphaS = np.nanmean(newrunsalphaS, axis=0)
plt.plot(mean_omegaS, mean_alphaS, 'k', linewidth=2)

plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.gca().set_aspect('equal')
# plt.savefig("coevolution_dependence-x_cohesion-y_d0=" + str(d0) + "_rHS=" + str(rHS) + ".pdf",format='pdf')
plt.show()

# dependence-cohesion

newrunsdependence = []
newrunscohesion = []
 
for omegaSrun, alphaSrun in zip(allrunsOmegaS, allrunsalphaS):
      newomega, newalpha = HomogeniseLength(omegaSrun, alphaSrun, maxruntime)
      plt.plot(omegaSrun, alphaSrun, '0.55', linewidth=0.65)

      newrunsOmegaS.append(newomega)
      newrunsalphaS.append(newalpha)

mean_omegaS = np.nanmean(newrunsOmegaS, axis=0)
mean_alphaS = np.nanmean(newrunsalphaS, axis=0)
plt.plot(mean_omegaS, mean_alphaS, 'k', linewidth=2)

plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.gca().set_aspect('equal')
# plt.savefig("coevolution_dependence-x_cohesion-y_d0=" + str(d0) + "_rHS=" + str(rHS) + ".pdf",format='pdf')
plt.show()


#      dependence  = omegaH_traj+omegaS_traj
#      cohesion = alphaH_traj+alphaS_traj

#      plt.plot(dependence,cohesion, linewidth=0.75)
#      plt.suptitle("dependence-cohesion")

# plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
# plt.xlim(0,1.1)
# plt.ylim(0,1.1)
# plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
# plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
# plt.gca().set_aspect('equal')
# # plt.savefig("coevolution_dependence-x_cohesion-y_d0=" + str(d0) + "_rHS=" + str(rHS) + ".pdf",format='pdf')
# plt.show()

