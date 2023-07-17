import numpy as np
import os
import matplotlib.pyplot as plt

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

# parameters needed for file

rH=8.0
rS=20.0
rHS=10.0

d0=50.0

time_skip=10
runlengths = []
allrunsH = []
allrunsS = []

# Read each file and plot it.


for fil in os.listdir('./results/data/'+ 'rHS=' + str(rHS) + '_d0=' + str(d0)):
     filename = './results/data/'+ 'rHS=' + str(rHS) + '_d0=' + str(d0) + '/' + fil
     trait_trajectoryH = np.loadtxt(filename, dtype='float', delimiter=',', skiprows=1, usecols=0)
     trait_trajectoryS = np.loadtxt(filename, dtype='float', delimiter=',', skiprows=1, usecols=1)

     trait_trajectoryH = trait_trajectoryH[0::time_skip]
     trait_trajectoryS = trait_trajectoryS[0::time_skip]

     runlengths.append(len(trait_trajectoryH))
     allrunsH.append(trait_trajectoryH)
     allrunsS.append(trait_trajectoryS)

     plt.plot(trait_trajectoryH, trait_trajectoryS, '0.55', linewidth=0.75) 

maxruntime = np.max(runlengths)
newrunsH = []
newrunsS = []

for runH, runS in zip(allrunsH, allrunsS):
      newH, newS = HomogeniseLength(runH, runS, maxruntime)

      newrunsH.append(newH)
      newrunsS.append(newS)

meanH = np.nanmean(newrunsH, axis=0)
meanS = np.nanmean(newrunsS, axis=0)
plt.plot(meanH, meanS, 'k', linewidth=2)

plt.plot(np.arange(0,1.0,0.001), np.arange(0,1.0,0.001), 'k', linestyle='--')
plt.plot(np.arange(0,1.0,0.001), [1.0]*len(np.arange(0,1.0,0.001)), 'k', linewidth=0.5)
plt.plot([1.0]*len(np.arange(0,1.0,0.001)), np.arange(0,1.0,0.001), 'k', linewidth=0.5)
plt.xlim(0,1.1)
plt.gca().set_aspect('equal')
plt.ylim(0,1.1)
plt.savefig("results/plots/stickiness-trajectory_rHS="+str(rHS) + "_d0=" + str(d0) +".pdf",format='pdf')
plt.show()

for trait_trajectoryH in allrunsH:
      plt.plot(range(len(trait_trajectoryH)), trait_trajectoryH, '0.55', linewidth=0.75)
      plt.plot(range(maxruntime), meanH, 'k', linewidth=2)

plt.plot(range(maxruntime), [1.0]*maxruntime, 'k', linestyle='--')
plt.ylim(0,1.1)
plt.savefig("results/plots/host-stickiness-trajectory_rHS="+str(rHS) + "_d0=" + str(d0) +".pdf",format='pdf')
plt.show()




