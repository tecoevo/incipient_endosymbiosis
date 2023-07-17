import numpy as np 
import matplotlib.pyplot as plt

variation_fH = True # do you want to visualise the variation of the bound with respect to a?
variation_CH = True # or with respect to d?

a = 2.1
d = 50.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
fS = 30.0
rHS = 10.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.

delta = 0.01
xrange = np.arange(0.0, 1.0, delta)
yrange = np.arange(0.0, 1.0, delta)
X, Y = np.meshgrid(xrange,yrange)

if variation_fH:
      CH=100.0
      for fH in np.arange(2.0, 50.0, 5): 
           F = d*(1-X*Y)
           G = rHS*X*Y*(1+a*np.sqrt((2*CH**2)/(fH*fS)))
           plt.contour(X, Y, (F - G), [0], alpha=fH/50.0)

      plt.xlim([0,1.05])
      plt.ylim([0,1.05])
      bounding_box = np.arange(0.0,1.1,0.1)
      plt.plot(bounding_box, [1.]*len(bounding_box), '--', color='k')
      plt.plot([1.]*len(bounding_box), bounding_box, '--', color='k')
      plt.savefig("feasibility-bound-different-fH.pdf",format='pdf')
      plt.show()

if variation_CH:
      d = 50.0
      for CH in np.arange(40, 400, 60): 
           F = d*(1-X*Y)
           G = rHS*X*Y*(1+a*np.sqrt((2*CH**2)/(fH*fS)))
           plt.contour(X, Y, (F - G), [0], alpha=CH/400.0)

      plt.xlim([0,1.05])
      plt.ylim([0,1.05])
      bounding_box = np.arange(0.0,1.1,0.1)
      plt.plot(bounding_box, [1.]*len(bounding_box), '--', color='k')
      plt.plot([1.]*len(bounding_box), bounding_box, '--', color='k')
      plt.savefig("feasibility-bound-different-CH.pdf",format='pdf')
      plt.show()