import numpy as np
import matplotlib.pyplot as plt
import os

rH=8.0
rS=20.0
rHS=10.0
d0=50.0

srcdir = './results/data/' # in which directory are the trajectory files that you want to analyse located?
                                        # make sure that this name starts with ./ and has / at the end

time_skip=10

allrunsOmegaH = []
allrunsalphaH = []
allrunsOmegaS = []
allrunsalphaS = []
runlengths = []

# record all trajectories
for fil in os.listdir(srcdir):
     if not fil.startswith('.'):
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
      plt.plot(range(len(omegaHrun)), omegaHrun, 'k')
      # plt.plot(range(len(alphaHrun)), alphaHrun, 'b')
      plt.xlim([0, maxruntime])
      # plt.plot(omegaHrun, alphaHrun, '0.55', linewidth=0.45)

      # newrunsOmegaH.append(newomega)
      # newrunsalphaH.append(newalpha)

# mean_omegaH = np.nanmean(newrunsOmegaH, axis=0)
# mean_alphaH = np.nanmean(newrunsalphaH, axis=0)
# plt.plot(mean_omegaH, mean_alphaH, 'k', linewidth=2)

# plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
# plt.xlim(0,1.1)
# plt.ylim(0,1.1)
# plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
# plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
# plt.gca().set_aspect('equal')
# plt.savefig("coevolution_omegaH-x_alphaH-y_d0=" + str(d0) + "_rHS=" + str(rHS) + ".pdf",format='pdf')
plt.savefig("omegaH-over-time.pdf", format='pdf')
plt.show()

# # symbiont traits
newrunsOmegaS = []
newrunsalphaS = []
 
for omegaSrun, alphaSrun in zip(allrunsOmegaS, allrunsalphaS):
      plt.plot(range(len(omegaSrun)), omegaSrun, 'k')
      # plt.plot(range(len(alphaSrun)), alphaSrun, 'b')
      plt.xlim([0, maxruntime])

plt.savefig("omegaS-over-time.pdf", format='pdf')
plt.show()

for alphaHrun in allrunsalphaH:
      plt.plot(range(len(alphaHrun)), alphaHrun, 'k')
      # plt.plot(range(len(alphaSrun)), alphaSrun, 'b')
      plt.xlim([0, maxruntime])

plt.savefig("alphaH-over-time.pdf", format='pdf')
plt.show()

for alphaSrun in allrunsalphaS:
      plt.plot(range(len(alphaSrun)), alphaSrun, 'k')
      # plt.plot(range(len(alphaSrun)), alphaSrun, 'b')
      plt.xlim([0, maxruntime])

plt.savefig("alphaS-over-time.pdf", format='pdf')
plt.show()

for alphaSrun in allrunsalphaS:
      plt.plot(range(len(alphaSrun)), alphaSrun, 'k')
      # plt.plot(range(len(alphaSrun)), alphaSrun, 'b')
      plt.xlim([0, maxruntime])

plt.savefig("alphaS-over-time.pdf", format='pdf')
plt.show()

for omegaHrun, omegaSrun, alphaHrun, alphaSrun in zip(allrunsOmegaH, allrunsOmegaS, allrunsalphaH, allrunsalphaS):
      dependencerun = np.array(omegaHrun)*np.array(omegaSrun)
      cohesionrun = np.array(alphaHrun)*np.array(alphaSrun)
      plt.plot(range(len(dependencerun)), dependencerun, 'b', label='dependence')
      plt.plot(range(len(cohesionrun)), cohesionrun, 'r', label='cohesion') 

plt.savefig("dependence-cohesion-over-time.pdf", format='pdf')
plt.show()


# for omegaHrun, alphaHrun in zip(allrunsOmegaH, allrunsalphaH):
#       plt.plot(omegaHrun, alphaHrun, alpha=0.55, lw=0.55, color='b')

# plt.savefig("omegaH-alphaH.pdf", format='pdf')
# plt.show()

# for alphaHrun, alphaSrun in zip(allrunsalphaH, allrunsalphaS):
#       plt.plot(alphaHrun, alphaSrun, alpha=0.55, lw=0.55, color='r')

# plt.savefig("alphaH-alphaS.pdf", format='pdf')
# plt.show()

# mean_omegaS = np.nanmean(newrunsOmegaS, axis=0)
# mean_alphaS = np.nanmean(newrunsalphaS, axis=0)
# plt.plot(mean_omegaS, mean_alphaS, 'k', linewidth=2)

# plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
# plt.xlim(0,1.1)
# plt.ylim(0,1.1)
# plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
# plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
# plt.gca().set_aspect('equal')
# # plt.savefig("coevolution_omegaS-x_alphaS-y_d0=" + str(d0) + "_rHS=" + str(rHS) + ".pdf",format='pdf')
# plt.show()

# # dependence-cohesion

# newrunsdependence = []
# newrunscohesion = []
 
# for omegaHrun, omegaSrun, alphaHrun, alphaSrun in zip(allrunsOmegaH, allrunsOmegaS, allrunsalphaH, allrunsalphaS):
#        dependencerun = np.array(omegaHrun)*np.array(omegaSrun)
#        cohesionrun = np.array(alphaHrun)*np.array(alphaSrun)
#        newdep, newcoh = HomogeniseLength(dependencerun, cohesionrun, maxruntime)
#        plt.plot(newdep, newcoh, '0.55', linewidth=0.45)

#        newrunsdependence.append(newdep)
#        newrunscohesion.append(newcoh)

# mean_dependence = np.nanmean(newrunsdependence, axis=0)
# mean_cohesion = np.nanmean(newrunscohesion, axis=0)
# plt.plot(mean_dependence, mean_cohesion, 'k', linewidth=2)

# plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
# plt.xlim(0,1.1)
# plt.ylim(0,1.1)
# plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
# plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
# plt.gca().set_aspect('equal')
# # plt.savefig("coevolution_dependence-x_cohesion-y_d0=" + str(d0) + "_rHS=" + str(rHS) + ".pdf",format='pdf')
# plt.show()

