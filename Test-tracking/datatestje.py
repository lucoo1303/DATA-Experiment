import scipy.odr as odr
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
import os

loc = os.path.dirname(__file__)
os.chdir(loc)
fname = 'output.txt'
print(fname)

data = np.genfromtxt(fname, delimiter='\t')

x = data[:,0]
y = data[:,1]


plt.figure()
plt.plot(x,'k.')
plt.title('x-positie met offset (gelezen) (pixel)')
plt.show()
