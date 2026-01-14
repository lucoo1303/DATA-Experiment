#%% -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:30:20 2018
Update 19-12-2023, Peter van Capel 
Update 11-12-2024, Peter van Capel
Update 19-12-2024, Boris van Boxtel en Peter van Capel
Update 08-10-2025, Peter van Capel
    In de nieuwste versie wordt gebruik gemaakt van trackingroutines die standaard 
    ingebouwd zijn in OpenCV. Hierdoor is de tracking betrouwbaarder en sneller 
    geworden, en dit script gebruiksvriendelijker. Nadeel is dat de trackingroutine
    tot op hele pixelposities een bepaling doet. In een volgende versie proberen we
    pixelinterpolatie in te bouwen. 

Tracking bij DATA-E Opdracht W56_CAMS
@author: capel102

Tip: evalueer het script in stappen (tussen de #%%-strepen) met knop pijltje + frame 
of Ctrl + Enter zodat je per stap kunt testen.
"""

# Laden packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#%% Algemene instellingen (zelf aan te passen)

# Als je tussentijds trackingresultaten en trackingoutput wilt laten zien, 
# dan zet je show op 1. Zonder plotten is sneller, maar met plotten geeft 
# de gelegenheid om tracking te checken (handig in de testfase)
show = 1

### Instellen bestandslocatie
# loc = 'C:/Temp' # Kan eventueel handmatig
# Onderstaande regels zetten de hoofddirectory gelijk aan de locatie van dit script
loc = os.path.dirname(__file__)
os.chdir(loc)

# Locatie video
directory = ''
# Naam video
#vidname = 'slinger.MOV' # Exilim
#vidname = 'slinger.MP4' # GoPro
vidname = 'testvid1.MOV' # Exilim

# Directory om resultaten weg te schrijven 
dir_write = ''
# Dit is de naam van het databestand waar de locaties van het object worden weggeschreven
filename_data = 'output.txt'
# Deze tekst wordt in de header van je databestand gezet
# Zo heb je context bij de inhoud van het databestand
header_text = 'Exilim, slinger, 240 fps'

#%% Beschikbare trackers (OpenCV 4.13)

# Trackers worden in volgende link kort beschreven
# https://docs.opencv.org/4.x/javadoc/org/opencv/video/Tracker.html
trackerTypes = ['CSRT', 'DaSiamRPN', 'GOTURN','KCF', 'MIL', 'Nano', 'Vit']
 
def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerCSRT_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerDaSiamRPN_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerNano_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerVit_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
 
  return tracker

# CSRT en KCF blijken in tests snel en betrouwbaar
tracker = createTrackerByName('CSRT')

#%% Openen video en plotten eerste frame

# Video-locatie
video = directory + vidname
# Open video-kanaal
cap = cv2.VideoCapture(video)

# Meta-informatie
# Total aantal frames
Nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Frames per second (fps) !!! niet correct for Exilim, stel handmatig in
fps = cap.get(cv2.CAP_PROP_FPS)
#fps = 240

### Analyse eerste frame
# Found is een boolean die aangeeft of het frame er is, frame zelf is BGR-data
found,frame = cap.read()
if not found:
  print('Failed to read video')

# Dimensies van array (ny rij, nx kolom)
(ny,nx,nc) = np.array(frame).shape

# Plot van eerste frane voor calibratie (wordt voor de zekerheid ook weggeschreven)
plt.figure('Calibratie')
plt.imshow(np.flip(frame,axis=2))
#plt.savefig('frame1.pdf')
plt.savefig('frame1.png',dpi=300)

#%% Selectie van object

bboxes = []
'''
Bepaal de bounding box van het object. Tip: trek de bounding box strak 
om het object, maar met enige marge zodat het hele object in de box past.
Door een willekeurige toets in te drukken (bv Enter) wordt de bounding box 
vastgelegd. 
'''
bbox = cv2.selectROI('Positie object', frame)
bboxes.append(list(bbox))

#%% Hier vindt de eigenlijke tracking plaats

cv2.destroyWindow("Positie object")

tracker.init(frame,bbox)
nframe = 0

def draw_bounding_box(bounding_box, frame):
    x, y, w, h = bounding_box
    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2, 2)
    return frame

while cap.isOpened():
    print('frame ' + str(nframe))
    bboxes.append(list(bbox))
    
    found, frame = cap.read()
    if not found:
        break
    
    found_object, bbox = tracker.update(frame)
    if show == 1:
        frame = draw_bounding_box(bbox, frame)
        cv2.imshow("Object location", frame)
        cv2.waitKey(1)
    nframe += 1  


# Alternatief voor plotten
# if show==1:
#     h.set_data(mask)
#     # Redraw
#     plt.draw()
#     # Pause om tijd te creëren voor update
#     plt.pause(0.1)

#%% Resultaten berekenen, plotten en wegschrijven

# Omrekenen bounding box naar x- en y-positie van midden van bounding box
pos = np.zeros((len(bboxes),2))
for n in np.arange(len(bboxes)):
    pos[n,0] = bboxes[n][0] + bboxes[n][2]/2
    pos[n,1] = bboxes[n][1] + bboxes[n][3]/2

### Plots van resultaten (je kunt deze met plt.savefig ook wegschrijven voor in je labjournaal)
plt.figure()
plt.plot(pos[:,0],'k.')
plt.title('x-positie (pixel)')
plt.vlines(x=[90,770], ymin=0, ymax=500)
plt.show()

plt.figure()
plt.plot(pos[:,1],'k.')
plt.title('y-positie (pixel)')
plt.show()

plt.figure()
plt.plot(pos[:,0],pos[:,1],'k.')
plt.xlim(0,nx)
plt.ylim(ny,0)
plt.title('y-positie tegen x-positie (pixel)')
plt.show()

"""
Tips:
    1. Je kunt er hier al voor kiezen om niet alle punten weg te schrijven maar 
    alleen de relevante punten (bijvoorbeeld pos[200:] schrijft alles vanaf frame 
    200 weg). Dat scheelt later weer in de verwerking. Kies wel dezelfde selectie 
    voor alle objecten, anders corresponderen de framenummers niet meer.
    2. Als je het dat-bestand met WordPad opent, kun je de filestructuur zien.
    Op elke rij staat de positie in elk frame.
"""

np.savetxt(dir_write + filename_data,pos[90:770],delimiter='\t',newline='\n',
           header=header_text)
