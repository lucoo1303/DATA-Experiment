import scipy.odr as odr
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
from uncertainties import unumpy as unp
import os


# gemeten constanten met onzekerheden en tags
m = unc.ufloat(92.28, 0.01, 'massa gewichtje')*0.001 # massa van gewichtje in kg
M = unc.ufloat(1459.03, 0.07, 'massa slinger')*0.001 # massa van slinger in kg
l = unc.ufloat(94.625, 0.025, 'lengte gewichtje')*0.01 # lengte tot zwaartepunt van gewichtje in m
L = unc.ufloat(102.05, 0.05, 'lengte slinger')*0.01 # lengte van slinger in m
h_veer = unc.ufloat(89.25, 0.05, 'hoogte veer')*0.01 # lengte tot veer bevestiging in m
m_veer = unc.ufloat(38.13, 0.45, 'massa aan veer')*0.001 # massa hangend aan veer in kg
u = unc.ufloat(3.85, 0.11, 'verplaatsing')*0.01 # verplaatsing van massa hangend aan veer onder invloed van zwaartekracht in m
g = unc.ufloat(9.80665, 0.00001, 'zwaartekrachtsversnelling') # zwaartekrachtsversnelling in m/s^2, met verwaarloosbare onzekerheid (maar niet nul, aangezien dat problemen kan veroorzaken in het onzekerhedenpakket)

# constanten afgeleid uit andere constanten met onzekerheden
I = m*l**2 + (M*L**2)/3
k = unc.ufloat((m_veer*g/u).n, (m_veer*g/u).s, 'veerconstante')
# Nieuwe ufloat gemaakt voor k, aangezien ik de foutpropagatie van m_veer en u niet zal bijhouden in het uiteindelijke experiment

# theoretische hoeksnelheid met onzekerheden
w1_theorie = unp.sqrt((m*l*g + M*L*g/2)/I)

# Functie om data per video te scheiden
def get_data_per_video(raw_data):
    seperated_data = []
    video_data = []
    for i in range(len(raw_data)):
        frame_no = raw_data[i][0]
        if frame_no == 0 and i != 0: # Nieuw filmpje begonnen, want frame nummer is 0
            seperated_data.append(np.array(video_data))
            video_data = []
        video_data.append(raw_data[i])
    video_data = np.array(video_data)
    seperated_data.append(video_data)
    return seperated_data





loc = os.path.dirname(__file__)
os.chdir(loc)
fname = 'output.txt'

raw_data = np.genfromtxt(fname, delimiter='\t')

data_per_video = get_data_per_video(raw_data)

data_vid0 = data_per_video[0]
data_vid1 = data_per_video[1]

frame_no_vid0 = data_vid0[:,0]
x_vid0 = data_vid0[:,1]
y_vid0 = data_vid0[:,2]

plt.figure()
plt.plot(frame_no_vid0, x_vid0, 'k.')
plt.title('x-positie met offset (gelezen) (pixel)')
plt.show()
