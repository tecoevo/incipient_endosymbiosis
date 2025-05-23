import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.integrate import odeint 
from matplotlib.lines import Line2D

which= 'omega' # allowed: omega or alpha 

# parameters that seem reasonable

CH = 100.0
CS = 200.0 # we assume that the carrying capacity of the symbiont is twice that of the host
CHS = 250.0
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.
a = 0.1 

dH = 0.0
dS = dH

bH = 1.0 
bS = bH
       
# initial conditions
xH0 = 10.0            # initial host abundance
xS0 = 10.0            # initial symbiont population
xHS0 = 0.0            # initial HS population
y0 = [xH0, xS0, xHS0]       # initial condition vector

# stability over omegas
if which == 'omega':
       rH = 8.0
       rS = 20.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
       rHS = 10.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
       d = 50.0

       omegaH = 0.0
       omegaS = 1.0

       dt = 0.0001
       xH = 0.0
       xS = 1.0
       xHS = 4.0
       xHarray = []
       xSarray = []
       xHSarray = []
       steps = 6000000
       for t in range(steps):
             xHarray.append(xH)
             xSarray.append(xS)
             xHSarray.append(xHS)
             xH += dt*(rH*(1-omegaH)*xH*(1-xH/CH) - a*xH*xS + d*xHS + dS*xHS + bH*xHS)
             xS += dt*(rS*(1-omegaS)*xS*(1-xS/CS) - a*xH*xS + d*xHS + dH*xHS + bS*xHS)
             xHS += dt*(rHS*omegaH*omegaS*xHS*(1-xHS/CHS) + a*xH*xS - d*xHS - dS*xHS - dH*xHS)
 
       print(xH, xS, xHS)

       plt.plot(range(steps), xHarray,label='Host')
       plt.plot(range(steps), xSarray,label='Symbiont')
       plt.plot(range(steps), xHSarray,label='Collective')
       plt.legend()
       plt.show()

if which == 'alpha':
       fH = 8.0
       fS = 20.0
       rHS = 40.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
       d0 = 50.0

       sigmaH = 0.2
       sigmaS = 0.2

       dt = 0.001
       xH = 10.0
       xS = 10.0
       xHS = 0.0
       xHarray = []
       xSarray = []
       xHSarray = []
       steps = 100000
       for t in range(steps):
             xHarray.append(xH)
             xSarray.append(xS)
             xHSarray.append(xHS)
             xH += dt*(fH*xH*(1-xH/CH) - a*xH*xS + d0*(1-sigmaH*sigmaS)*xHS + dS*xHS + bH*xHS)
             xS += dt*(fS*xS*(1-xS/CS) - a*xH*xS + d0*(1-sigmaH*sigmaS)*xHS + dH*xHS + bS*xHS)
             xHS += dt*(rHS*sigmaH*sigmaS*xHS*(1-xHS/CHS) + a*xH*xS - d0*(1-sigmaH*sigmaS)*xHS - dS*xHS - dH*xHS)
 
       print(xH, xS, xHS)

       plt.plot(range(steps), xHarray,label='Host')
       plt.plot(range(steps), xSarray,label='Symbiont')
       plt.plot(range(steps), xHSarray,label='Collective')
       plt.legend()
       plt.show()
