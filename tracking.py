# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 11:51:00 2025

Track multiple objects in the same video. 

Version 07-01-2026
This script is under development. Only CSRT tracker is possible and it is not 
yet possible to display live tracking results. 

@author: capel102
"""

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
 

#%%

# Trackers
# https://docs.opencv.org/4.x/javadoc/org/opencv/video/Tracker.html

from random import randint
 
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
 
def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
 
  return tracker

#%%

### Instellen bestandslocatie
# loc = 'C:/Temp' # Kan eventueel handmatig
# Onderstaande regels zetten de hoofddirectory gelijk aan de locatie van dit script
loc = os.path.dirname(__file__)
os.chdir(loc)

# Locatie video
directory = 'data/'
# Naam video
#vidname = 'CIMG1828.MOV' # Exilim
vidname = 'slinger.MP4' # GoPro

# Directory om resultaten weg te schrijven 
dir_write = 'output/'
# Dit is de naam van het databestand waar de locaties van het object worden weggeschreven
filename_data = 'slinger.dat'
# Deze tekst wordt in de header van je databestand gezet
# Zo heb je context bij de inhoud van het databestand
header_text = 'GoPro, slinger, 240 fps'

 
# Create a video capture object to read videos
cap = cv2.VideoCapture(directory + vidname)
 
# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)
  
Nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
# Dimensies van array (ny rij, nx kolom)
(ny,nx,nc) = np.array(frame).shape


#%% Number of trackers

# Decide on how many objects to track
nobj = 3

## Select boxes
bboxes = np.zeros((nobj,4),dtype=int)
colors = [] 

for n in range(nobj):
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    bboxes[n] = cv2.selectROI('MultiTracker'+str(n), frame)
    exec('obj'+str(n) +' = []')

#%% Create trackers and perform tracking

for n in range(nobj):
    exec('tracker'+str(n) +' = cv2.TrackerCSRT_create()')
    exec('tracker'+str(n) +'.init(frame, bboxes['+str(n)+'])')

# Array to collect all bboxes
pos_all = []
# Initial positions
for n in range(nobj):
    pos_all.append(bboxes[n].tolist())

# # Process video and track objects
nframe = 1
while cap.isOpened():
    print('frame ' + str(nframe))
    
    success, frame = cap.read()
    if not success:
        print('End tracking')
        break
  
    # get updated location of objects in subsequent frames
    for n in range(nobj):
        exec('found,bboxes['+str(n)+'] = tracker'+str(n) +'.update(frame)')
        #print(bboxes[n])
        pos_all.append(bboxes[n].tolist())
    nframe += 1  

#%% Convert bboxes to positions for each object

# Create location array
pos = np.zeros((Nframes, 2*nobj))

# Determine positions from bounding box
for n in np.arange(len(pos_all)):
    pos[int(np.floor(n/3)),2*np.mod(n,3)] = pos_all[n][0] + pos_all[n][2]/2
    pos[int(np.floor(n/3)),2*np.mod(n,3) + 1] = pos_all[n][1] + pos_all[n][3]/2

"""
Array pos contains 2*nobj columns of length Nframes, where column 2*i is the 
x position of object i and 2*i + 1 the y position.
"""
    
#%% Write and plot basic results
    
# Directory om tussenresultaten weg te schrijven 
dir_write = 'output/'

### Plots van resultaten (je kunt deze met plt.savefig ook wegschrijven voor in je labjournaal)
plt.figure()
for n in np.arange(nobj):
    plt.plot(pos[:,2*n], marker='.', label='Object ' + str(n))
plt.xlabel('Frame')
plt.ylabel('x-positie (pixel)')
plt.legend()
plt.show()
plt.savefig(dir_write + 'slinger' + '_x.png', dpi=300)

plt.figure()
for n in np.arange(nobj):
    plt.plot(pos[:,2*n + 1], marker='.', label='Object ' + str(n))
plt.xlabel('Frame')
plt.ylabel('y-positie (pixel)')
plt.legend()
plt.show()
plt.savefig(dir_write + 'slinger' + '_y.png', dpi=300)


plt.figure()
for n in np.arange(nobj):
    plt.plot(pos[:,2*n],pos[:,2*n + 1], marker='.', label='Object ' + str(n))
plt.ylabel('x-positie (pixel)')
plt.ylabel('y-positie (pixel)')
plt.xlim(0,nx)
plt.ylim(ny,0)
plt.title('y-positie tegen x-positie (pixel)')
plt.legend()
plt.show()
plt.savefig(dir_write + 'slinger' + '_xy.png', dpi=300)

"""
Tips:
    1. Je kunt er hier al voor kiezen om niet alle punten weg te schrijven maar 
    alleen de relevante punten (bijvoorbeeld pos[200:] schrijft alles vanaf frame 
    200 weg). Dat scheelt later weer in de verwerking. Kies wel dezelfde selectie 
    voor alle objecten, anders corresponderen de framenummers niet meer.
    2. Als je het dat-bestand met WordPad opent, kun je de filestructuur zien.
    Op elke rij staat de positie in elk frame.
"""

np.savetxt(dir_write + filename_data,pos,delimiter='\t',newline='\n',
           header=header_text)