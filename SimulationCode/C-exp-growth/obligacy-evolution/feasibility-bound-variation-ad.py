import numpy as np 
import matplotlib.pyplot as plt

variation_a = True # do you want to visualise the variation of the bound with respect to a?
variation_d = True # or with respect to d?

CH = 100.0
CS = 2*CH # we assume that the carrying capacity of the symbiont is twice that of the host
rH = 8.0
rS = 30.0 # how large is the pure symbiont growth rate as compared to the pure host growth rate?
rHS = 10.0 # we assume that the complex, at its full performance, can grow at twice the rate of the host. 
# This is an assumption of the ecology - we visualise the case when the symbiont allows the host to occupy a new niche.

delta = 0.01
xrange = np.arange(0.0, 1.0, delta)
yrange = np.arange(0.0, 1.0, delta)
X, Y = np.meshgrid(xrange,yrange)

if variation_d:
      a = 2.1
      for d in np.arange(25, 750, 100): 
           F = d
           G = rHS*X*Y*(1+a*np.sqrt((CH*CS)/(rH*rS*(1-X)*(1-Y))))
           plt.contour(X, Y, (F - G), [0], alpha=d/750)

      plt.xlim([0,1.05])
      plt.ylim([0,1.05])
      bounding_box = np.arange(0.0,1.1,0.1)
      plt.plot(bounding_box, [1.]*len(bounding_box), '--', color='k')
      plt.plot([1.]*len(bounding_box), bounding_box, '--', color='k')
      plt.gca().set_aspect('equal')
      plt.savefig("feasibility-bound-different-d.pdf",format='pdf')
      plt.show()

if variation_a:
      d = 50.0
      for a in np.arange(0.01, 5.0, 0.75): 
           F = d
           G = rHS*X*Y*(1+a*np.sqrt((CH*CS)/(rH*rS*(1-X)*(1-Y))))
           plt.contour(X, Y, (F - G), [0], alpha=a/5.0)

      plt.xlim([0,1.05])
      plt.ylim([0,1.05])
      bounding_box = np.arange(0.0,1.1,0.1)
      plt.plot(bounding_box, [1.]*len(bounding_box), '--', color='k')
      plt.plot([1.]*len(bounding_box), bounding_box, '--', color='k')
      plt.gca().set_aspect('equal')
      plt.savefig("feasibility-bound-different-a.pdf",format='pdf')
      plt.show()