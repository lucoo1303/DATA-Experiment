import scipy.odr as odr
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
from uncertainties import unumpy as unp
import os
import warnings

warnings.filterwarnings("ignore")


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

# Camera eigenschappen:
fps = 240
pixel_width = 512 # in pixels
pixel_pos_unc = 1.0 # onzekerheid in pixel positie 
frame_unc = 0.5 # onzekerheid in frame nummer 
frame_width = unc.ufloat(68.2, 0.1, 'frame breedte')*0.01 # in m
pixel_scale = frame_width / pixel_width # m per pixel

# Functie om uitwijking (in x-positie) om te zetten naar hoek in radialen
def delection_to_angle(deflection, length):
    return unp.arcsin(deflection / length)

# Functie voegt onzekerheden toe aan data dmv een uarray
def add_uncertainties_to_data(data, sig_f, sig_x, sig_y):
    frame_u = unp.uarray(data[:, 0], sig_f)
    x_u = unp.uarray(data[:, 1], sig_x)
    y_u = unp.uarray(data[:, 2], sig_y)
    return np.column_stack((frame_u, x_u, y_u))

# Functie zet video data om naar fysieke data
def convert_video_data_to_physical(data, pixel_scale, fps):
    t = convert_frame_to_time(data[:,0], fps)
    x = convert_pixel_to_physical(data[:,1], pixel_scale)
    y = convert_pixel_to_physical(data[:,2], pixel_scale)
    return np.column_stack((t, x, y))
    
# Functie zet frame nummer om naar tijd op basis van fps
def convert_frame_to_time(frame_no, fps):
    return frame_no / fps

# Functie zet pixelpositie om naar fysieke positie op basis van schaal
def convert_pixel_to_physical(pixel_value, pixel_scale):
    return pixel_value * pixel_scale

# Functie om data per video te scheiden
def get_data_per_video(data):
    seperated_data = []
    video_data = []
    for i in range(len(data)):
        frame_no = data[i][0]
        if frame_no.n == 0 and i != 0: # Nieuw filmpje begonnen, want frame nummer is 0
            seperated_data.append(np.array(video_data))
            video_data = []
        video_data.append(data[i])
    video_data = np.array(video_data)
    seperated_data.append(video_data)
    return seperated_data

def add_angle_data(data, length):
    angle = delection_to_angle(data[:,1], length)
    return np.column_stack((data, angle))

# Functie om de data van een specifieke video te plotten
def plot_vid_data(data, vid_num):
    vid = data[vid_num]
    t = vid[:,0]
    x = vid[:,1]
    y = vid[:,2]
    theta = vid[:,3]

    # de nominale waardes
    t_n = unp.nominal_values(t)
    x_n = unp.nominal_values(x)
    y_n = unp.nominal_values(y)
    theta_n = unp.nominal_values(theta)

    # de onzekerheden
    t_s = unp.std_devs(t)
    x_s = unp.std_devs(x)
    y_s = unp.std_devs(y)
    theta_s = unp.std_devs(theta)

    plt.figure()
    plt.errorbar(t_n, x_n, xerr=t_s, yerr=x_s, fmt='.')
    plt.title(f'Video {vid_num}: x-positie op tijdstip')
    plt.xlabel('tijd (s)')
    plt.ylabel('x positie (m)')
    plt.show()

    plt.figure()
    plt.errorbar(t_n, y_n, xerr=t_s, yerr=y_s, fmt='.')
    plt.title(f'Video {vid_num}: y-positie op tijdstip')
    plt.xlabel('tijd (s)')
    plt.ylabel('y positie (m)')
    plt.show()

    plt.figure()
    plt.errorbar(x_n, y_n, xerr=x_s, yerr=y_s, fmt='.')
    plt.title(f'Video {vid_num}: y positie tegen x positie')
    plt.xlabel('x positie (m)')
    plt.ylabel('y positie (m)')
    plt.show()

    plt.figure()
    plt.errorbar(x_n, theta_n, xerr=x_s, yerr=theta_s, fmt='.')
    plt.title(f'Video {vid_num}: hoek tegen x positie')
    plt.xlabel('x positie (m)')
    plt.ylabel('hoek (rad)')
    plt.show()

    plt.figure()
    plt.errorbar(t_n, theta_n, xerr=t_s, yerr=theta_s, fmt='.')
    plt.title(f'Video {vid_num}: hoek tegen tijd')
    plt.xlabel('tijd (s)')
    plt.ylabel('hoek (rad)')
    plt.show()


loc = os.path.dirname(__file__)
os.chdir(loc)
fname = 'output.txt'

# De data regelrecht uit het databestand
raw_data = np.genfromtxt(fname, delimiter='\t')
# De data met toegevoegde onzekerheden
raw_data_unc = add_uncertainties_to_data(raw_data, frame_unc, pixel_pos_unc, pixel_pos_unc)
# De data omgezet naar fysieke eenheden (dus frame nummer -> seconde, pixel -> meter)
phys_data = convert_video_data_to_physical(raw_data_unc, pixel_scale, fps)
# Data with angle
angle_data = add_angle_data(phys_data, l)
# De data gescheiden per video
data_per_video = get_data_per_video(angle_data)

plot_vid_data(data_per_video, 0)
#plot_vid_data(data_per_video, 1)
