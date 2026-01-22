#-*- coding: utf-8 -*-
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

# Functie om de schatting van het loslaatmoment te plotten ter controle
def plot_index_of_release(x_pos, index):
  plt.figure()
  plt.plot(x_pos,'k.')
  plt.title('onbewerkte x-positie (pixel)')
  plt.vlines(x=index, ymin=-200, ymax=200)
  plt.show()

# Schat ruwweg het moment waarop de slinger in de video wordt losgelaten
def get_index_of_release(x_pos):
  # Vind de maximale (start)uitwijking
  abs_max = np.max(np.abs(x_pos))
  index = 0
  # Doorloop de video tot het punt waarop de uitwijking ongeveer gelijk is aan de max
  while np.abs(x_pos[index]) < 0.95*abs_max:
      index += 1
  # De slinger zit nu ongeveer op de startpositie
  jump = 10
  # Doorloop nu de video totdat de uitwijking ietsjes later in de video
  # signifant kleiner is, dat is ongeveer het punt van loslaten
  while np.abs(x_pos[index+jump]) > 0.95*np.abs(x_pos[index]) and index < len(x_pos) - jump - 1:
      index += 1
  # Checken of de schatting klopt
  index_correct = False
  while not index_correct:
    plot_index_of_release(x_pos, index)
    user_input = input(f'Is index of release {index} correct? y/n ')
    if user_input == 'y':
        index_correct = True
    else:
        index = int(input('Geef de correcte index of release op: '))
  return index

# functie om de header aan te maken voor het databestand
def create_header(video_name):
  header = 'Exilim, slinger, 240 fps, 512x384 px, 17-01-2026 \n'
  header += 'Video: ' + video_name + ' \n'
  header += 'Seperate video data is seperated with a new header \n'
  header += 'Uncertainty in frame number: approx. 2.85 ms, approx 1/2 frame \n'
  header += 'Uncertainty in position: 1 pixel \n'
  header += 'Columns: \n'
  header += 'frame number                x (px)                       y (px)'
  return header

# Functie om de data om te zetten naar standaard cartesische coordinaten, met
# de assen gecentreerd op de neutrale positie
def offset_data(data):
  x = data[:,0]
  y = 384 - data[:,1]
  return np.array([x - x[0], y - y[0]]).T

# Functie om de data te corrigeren en te slicen
def correct_and_slice_data(data):
  # Corrigeer de data naar simpelere assen
  pos_offset = offset_data(data)
  x_offset = pos_offset[:,0]
  y_offset = pos_offset[:,1]

  index_of_release = get_index_of_release(x_offset) # ongeveer frame 1800 bij testvid4.MOV

  # Pak alleen de relevante data (na loslaten slinger)
  x_data = x_offset[index_of_release:]
  y_data = y_offset[index_of_release:]
  frame_data = np.arange(len(x_data))

  return np.array([frame_data, x_data, y_data]).T, index_of_release

# functie om de onbewerkte data te plotten ter controle
def plot_uncorrected_data(data, index_of_release = 0, nx=512, ny=384):
  x = data[:,0]
  y = data[:,1]
  # Plots van onbewerkte data

  plt.figure()
  plt.plot(x,'k.')
  plt.title('onbewerkte x-positie (pixel)')
  plt.vlines(x=index_of_release, ymin=0, ymax=500)
  plt.show()

  plt.figure()
  plt.plot(y,'k.')
  plt.title('onbewerkte y-positie (pixel)')
  plt.show()

  plt.figure()
  plt.plot(x,y,'k.')
  plt.xlim(0,nx)
  plt.ylim(ny,0)
  plt.title('onbewerkte y-positie tegen onbewerkte x-positie (pixel)')
  plt.show()

# functie om de gecorrigeerde data te plotten ter controle
def plot_corrected_data(data):
  x_data = data[:,1]
  y_data = data[:,2]

  plt.figure()
  plt.plot(x_data,'k.')
  plt.title('gecorrigeerde x-positie (pixel)')
  plt.show()

  plt.figure()
  plt.plot(y_data,'k.')
  plt.title('gecorrigeerde y-positie (pixel)')
  plt.show()

  plt.figure()
  plt.plot(x_data,y_data,'k.')
  plt.ylim(-100, 100)
  plt.title('gecorrigeerde y-positie tegen x-positie (pixel)')
  plt.show()

# functie om data weg te schrijven naar een bestand
def write_data_to_file(data, header, output_path):
  # wegschrijven na check data
  # Checken of data goed gesliced is voor wegschrijven, of dat ik opnieuw moet tracken
  # Te zien aan de vline in de onbewerkte x-positie plot
  is_data_correct = input("Data correct? y/n ")

  # Schrijf de data weg als deze goed is gesliced
  if is_data_correct == 'y':
      with open(output_path, 'a') as f:
          np.savetxt(f,data,delimiter='\t',newline='\n',header=header)

def track_video(vid_num, directory_video, output_path):
  # Algemene instellingen (zelf aan te passen)

  # Als je tussentijds trackingresultaten en trackingoutput wilt laten zien, 
  # dan zet je show op 1. Zonder plotten is sneller, maar met plotten geeft 
  # de gelegenheid om tracking te checken (handig in de testfase)
  show = 1

  ### Instellen bestandslocatie

  # Naam video
  #vidname = 'slinger.MP4' # GoPro
  video_name = f'vid{vid_num}.MOV' # Exilim

  # Deze tekst wordt in de header van je databestand gezet
  # Zo heb je context bij de inhoud van het databestand
  header = create_header(video_name)

  # Beschikbare trackers (OpenCV 4.13)

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

  # Openen video en plotten eerste frame

  # Video-locatie
  video = directory_video + video_name
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
  plt.savefig(f'frame1_{video_name}.png',dpi=300)

  # Selectie van object

  bboxes = []
  '''
  Bepaal de bounding box van het object. Tip: trek de bounding box strak 
  om het object, maar met enige marge zodat het hele object in de box past.
  Door een willekeurige toets in te drukken (bv Enter) wordt de bounding box 
  vastgelegd. 
  '''
  bbox = cv2.selectROI('Positie object', frame)
  bboxes.append(list(bbox))

  #Hier vindt de eigenlijke tracking plaats

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

  # data ophalen en snijden

  # Omrekenen bounding box naar x- en y-positie van midden van bounding box
  pos = np.zeros((len(bboxes),2))
  for n in np.arange(len(bboxes)):
      pos[n,0] = bboxes[n][0] + bboxes[n][2]/2
      pos[n,1] = bboxes[n][1] + bboxes[n][3]/2

  data, index_of_release = correct_and_slice_data(pos)

  # Plot de originele onbewerkte data ter controle
  plot_uncorrected_data(pos, index_of_release, nx, ny)

  # Plots van gecorrigeerde data ter controle
  plot_corrected_data(data)

  """
  Tips:
      1. Je kunt er hier al voor kiezen om niet alle punten weg te schrijven maar 
      alleen de relevante punten (bijvoorbeeld pos[200:] schrijft alles vanaf frame 
      200 weg). Dat scheelt later weer in de verwerking. Kies wel dezelfde selectie 
      voor alle objecten, anders corresponderen de framenummers niet meer.
      2. Als je het dat-bestand met WordPad opent, kun je de filestructuur zien.
      Op elke rij staat de positie in elk frame.
  """
  # Data wegschrijven naar bestand
  write_data_to_file(data, header, output_path)


def track_videos(video_indices, directory_video='Videos/', output_path='Data/output.txt'):
  verwijder_oude_data = input('Verwijder oude data? y/n ')
  # Eventuele oudere data verwijderen voordat er nieuwe wordt bijgeschreven
  if verwijder_oude_data == 'y':
      with open(output_path, "r+") as f:
          f.seek(0)
          f.truncate()
  # Elke video tracken
  for i in video_indices:
    print(f'Nu tracken: vid{i}.MOV')
    track_video(i, directory_video, output_path)
    print(f'Laatst getrackte video: vid{i}.MOV')


# Onderstaande regels zetten de hoofddirectory gelijk aan de locatie van dit script
loc = os.path.dirname(__file__)
os.chdir(loc)

# alle video's:
vid_amount = 25
indices = np.arange(1,vid_amount + 1)

# specifieke video's:
#indices = [3,5,8,9,10,11,12,15,24]

track_videos(indices, output_path='Data/output.txt')