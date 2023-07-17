import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

# define constants

fH = 10.0
fS = 20.0
fHS = 8.0
# fH, fS, fHS = [1.5299194165819079, 6.210242962900399, 0.5344473597037803]
# print(fH, fS, fHS)
kH = 2.0
kS = 0.01
a = 0.05
d = 50.0 
CH = 100.0
CS = 200.0

# initial conditions
xH0 = 1            # initial host abundance
xS0 = 1            # initial symbiont population
xHS0 = 1            # initial HS population
y0 = [xH0, xS0, xHS0]       # initial condition vector
t  = np.linspace(0, 20, 1000)       # time grid

# integrate ODEs 

def f(y, t):
     xH = y[0]
     xS = y[1]
     xHS = y[2]
     f0 = fH*xH*(1-xH/CH) + kS*xH*xS - a*xH*xS + d*xHS
     f1 = fS*xS*(1-xS/CS) +kH*xH*xS - a*xH*xS + d*xHS 
     f2 = fHS*xHS + a*xH*xS - d*xHS
     return [f0, f1, f2] 

soln = odeint(f, y0, t)
xH = soln[:, 0]
xS = soln[:, 1]
xHS = soln[:, 2]

print(xH[-1], xS[-1], xHS[-1])
plt.plot(t,xH,label='Host')
plt.plot(t,xS,label='Symbiont')
plt.plot(t,xHS,label='Host-symbiont association')
plt.legend()
plt.show()