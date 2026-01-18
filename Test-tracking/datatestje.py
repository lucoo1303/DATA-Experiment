import scipy.odr as odr
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
import os


# defining meassured constants with their uncertainties and tags for tracking purposes
m = unc.ufloat(92.28, 0.01, 'mass bob')*0.001 # mass of bob in kg
M = unc.ufloat(1459.03, 0.07, 'mass rod')*0.001 # mass of rod in kg
l = unc.ufloat(94.625, 0.025, 'length bob')*0.01 # length to center of mass of bob in m
L = unc.ufloat(102.05, 0.05, 'length rod')*0.01 # length of rod in m
h_spring = unc.ufloat(89.25, 0.05, 'spring height')*0.01 # length to spring attachment in m
m_spring = unc.ufloat(38.13, 0.45, 'mass on spring')*0.001 # mass hanging onto spring in kg
u = unc.ufloat(3.85, 0.11, 'displacement')*0.01 # displacement of mass hanging on spring under influence of gravity in m
g = unc.ufloat(9.80665, 0.00001, 'gravitational acceleration') # gravitational acceleration in m/s^2, with negligible uncertainty (but not zero, since that could cause issues in the uncertainty package)




loc = os.path.dirname(__file__)
os.chdir(loc)
fname = 'output.txt'

raw_data = np.genfromtxt(fname, delimiter='\t')
seperated_data = []
video_data = []
for i in range(len(raw_data)):
    frame_no = raw_data[i][0]
    if frame_no == 0 and i != 0:
        seperated_data.append(np.array(video_data))
        video_data = []
    video_data.append(raw_data[i])
video_data = np.array(video_data)
seperated_data.append(video_data)

vid0_data = seperated_data[0]
vid1_data = seperated_data[1]



frame_no = raw_data[:,0]
x = raw_data[:,1]
y = raw_data[:,1]

plt.figure()
plt.plot(frame_no, x, 'k.')
plt.title('x-positie met offset (gelezen) (pixel)')
plt.show()
