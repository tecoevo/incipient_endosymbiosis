import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from matplotlib.colors import TwoSlopeNorm, ListedColormap
import matplotlib.ticker as ticker

# See the script `simulate-popdyn.py` in the same directory if you want to solve the exponential model ODEs
# for a particular parameter combination.

vary_a = False
vary_fi = True

def GetMaxRe(M):
      " Get the maximum real part of the eigenvalues of the given matrix M. "
      eigenvalues = np.linalg.eigvals(M)
      return np.max([np.real(val) for val in eigenvalues])

# define constants

CH = 100.0
CS = 2*CH

#### We will calculate eigenvalues of the Jacobian and plot the maximum real part as a measure of stability 

# # We want to know what happens in two cases: 
# First we fix the intrinsic growth rates f_i and plot stability of the fixed point as a function of the
# association-dissociation rates a and d. These results will be stored in the matrix `ad_stability`. 
# Secondly we fix a and d using the knowledge from the first exercise and plot stability as a function
# of the intrinsic growth rates f_i. Here we will make some assumptions: 
# These results will be stored in the matrix `fi_stability`. 
# The carrying capacities are fixed at CH=20, CS = 2*CH throughout since in nature there are usually a lot
# more symbionts than hosts.

# Case 1: Fix f_i and vary a and d. 
# The rationale for the below values is that fS > fH because symbionts usually have smaller generation 
# times. Secondly, fHS > fH because we assume that the association of host with symbiont gives them
# some kind of selective advantage. We keep fH < fS because of the intuitive reason that the host timescale
# sets the collective timescale in cases like mitochondria, aphids, etc. 
# Lastly, we set the f_i such that they are small compared to the carrying capacities since the f_i are 
# interpreted as the number of offpsring per time unit, and it must intuitively take a nontrivial number
# of generations to reach the carrying capacity.

fH = 8.0
fS = 20.0
fHS = 10.0

if vary_a==True:
      min_a = 0.0
      max_a = 40.0
      step_a = 0.5
      a_params = np.arange(min_a,max_a,step_a)

      d_multiplier = 5.0

      ad_stability = np.zeros([len(a_params), len(a_params)]) 

      for i in range(len(a_params)):
            a = a_params[i]

            min_d = fHS*(1+max_a*CH/fH)
            max_d = d_multiplier*fHS*(1+max_a*CH/fH)
            step_d = (max_d-min_d)/float(len(a_params))
            d_params = np.arange(min_d, max_d, step_d)

            for j in range(len(d_params)):
                  d = d_params[j]
                  # calculate the equilibrium population abundances for these parameter values
                  if (CH*CS*(a*fHS)**2 - fH*((d - fHS)**2)*fS) == 0:
                        continue
                  xHstar = (CH*fS*(fHS-d)*((d-fHS)*fH + a*CS*fHS))/((a**2)*CH*CS*(fHS**2)-fH*fS*((d-fHS)**2))
                  xSstar = (CS*fH*(fHS-d)*((d-fHS)*fS + a*CH*fHS))/((a**2)*CH*CS*(fHS**2)-fH*fS*((d-fHS)**2))

                  jacobian = [[fH*(1-2*xHstar/CH) - a*xSstar, -a*xHstar, d],
                              [-a*xSstar, fS*(1-2*xSstar/CS) - a*xHstar, d],
                              [a*xSstar, a*xHstar, fHS-d]]

                  ad_stability[len(a_params)-1-i][j] = GetMaxRe(jacobian)

      colorbar_size = np.max([np.abs(np.min(ad_stability)), np.abs(np.max(ad_stability))])

      # plot results
      plt.imshow(ad_stability,interpolation='none', norm=TwoSlopeNorm(vmin=-colorbar_size, vcenter=0.0,
            vmax=colorbar_size), cmap=ListedColormap([[0,0,1],[1,0,0]]), aspect='auto',
            extent = [a_params[0], a_params[-1], 1.0, d_multiplier])

      plt.xlabel("Association rate, a")
      plt.ylabel("Dissociation rate, d, in multiples of the feasibility bound")
      plt.colorbar()
      plt.savefig("stability-interior-fpt_vary-ad_fHS="+ str(fHS) +".pdf", format='pdf')
      plt.show()
      plt.close()

##################################

##### Case 2: Fix a and d, vary f_i.
# Here we use the previous knowledge to set a and d. We fix fH < fS for the same reason as before, and
# vary fHS from 0 upto fS. This is motivated again by the fact that the host generation time seems to set
# the collective generation time. 

if vary_fi==True:
      max_fHS = 5*fS
      f_params = np.arange(0, max_fHS, 0.5)
      len_fparams = len(f_params)
      fi_stability = np.zeros([len_fparams]) # to store the value of maximum real part

      a = 0.1
      d = a+max_fHS*(1+a*np.sqrt((CH*CS)/(fH*fS))) # we want to make sure d is always large enough to ensure feasibility

      for i in range(len(f_params)):
            fHS = f_params[i]

            if (CH*CS*(a*fHS)**2 - fH*((d - fHS)**2)*fS) == 0:
                        continue
            
            # calculate the equilibrium population abundances for these parameter values
            xHstar = (CH*fS*(fHS-d)*((d-fHS)*fH + a*CS*fHS))/((a**2)*CH*CS*(fHS**2)-fH*fS*((d-fHS)**2))
            xSstar = (CS*fH*(fHS-d)*((d-fHS)*fS + a*CH*fHS))/((a**2)*CH*CS*(fHS**2)-fH*fS*((d-fHS)**2))

            jacobian = [[fH*(1-2*xHstar/CH) - a*xSstar, -a*xHstar, d],
                        [-a*xSstar, fS*(1-2*xSstar/CS) - a*xHstar, d],
                        [a*xSstar, a*xHstar, fHS-d]]

            fi_stability[i] = GetMaxRe(jacobian)

      # plot results
      plt.plot(f_params,fi_stability)
      plt.plot(f_params, [0.0]*len_fparams, 'k', '--')
      plt.yticks(np.arange(0, np.max(fi_stability), 0.5), np.arange(0, np.max(fi_stability), 0.5))

      # plt.savefig("stability-interior-fpt_vary-fHS_a="+str(a)+".pdf", format='pdf')
      plt.show()