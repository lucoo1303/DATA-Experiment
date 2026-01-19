import scipy.odr as odr
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
from uncertainties import unumpy as unp
import math
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
w1_theorie = ((m*l*g + M*L*g/2)/I)**0.5

# tijdstippen waar ik de data wil evalueren 
tn = np.arange(1, 6, 0.5)

# Camera eigenschappen:
fps = 240
pixel_width = 512 # in pixels
pixel_pos_unc = 1.0 # onzekerheid in pixel positie 
frame_unc = 0.5 # onzekerheid in frame nummer 
frame_width = unc.ufloat(68.2, 0.1, 'frame breedte')*0.01 # in m
pixel_scale = frame_width / pixel_width # m per pixel

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

# Functie om data per video te scheiden, returnt een lijst van numpy arrays
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

# Functie om uitwijking (in x-positie) om te zetten naar hoek in radialen
def deflection_to_angle(deflection, length):
    return unp.arcsin(deflection / length)

# Functie om hoek data toe te voegen aan dataset
def add_angle_data(data, length):
    angle = deflection_to_angle(data[:,1], length)
    return np.column_stack((data, angle))

# Functie om beginhoek uit data te halen
def get_theta0(data):
    return data[:,3][0]

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
    plt.errorbar(t_n, theta_n, xerr=t_s, yerr=theta_s, fmt='.')
    plt.title(f'Video {vid_num}: hoek tegen tijd')
    plt.xlabel('tijd (s)')
    plt.ylabel('hoek (rad)')
    plt.show()

# functie om dichtstbijzijnde index te vinden
def get_index_of_closest_val(data, target):
    vals = unp.nominal_values(data)
    index = np.argmin(np.abs(vals - target))
    return index

# functie om benaderings onzekerheid in hoek te schatten
def uncertaninty_in_angle_due_to_approximation(angles, index):
    angle = angles[index]
    prev_angle = angles[index - 1].n
    next_angle = angles[index + 1].n if index + 1 < len(angles) else angles[index].n
    diff1 = np.abs(angle.n - prev_angle)/2
    diff2 = np.abs(angle.n - next_angle)/2
    angle_unc = diff1 + diff2 + angle.s
    return angle_unc

# functie om hoek op een specifiek tijdstip te krijgen, geeft de tijd en hoek terug met bijgekomen benaderings onzekerheid
def theta_and_t_at_tn(data, target_time):
    # zoek dichtstbijzijnde tijdstip in data
    times = data[:,0]
    angles = data[:,3]
    i = get_index_of_closest_val(times, target_time)
    time_unc = times[i].s + np.abs(times[i].n - target_time)
    t = unc.ufloat(times[i].n, time_unc)
    angle_n = angles[i].n
    angle_s = uncertaninty_in_angle_due_to_approximation(angles, i)
    angle = unc.ufloat(angle_n, angle_s)
    
    noms = [t.n, angle.n]
    stds = [t.s, angle.s]
    return unp.uarray(noms, stds)

# functie om de data van alle video's op specifiek tijdstip tn te verzamelen
def data_tn(data, tn):
    data_at_tn = []
    for vid in data:
        theta0 = get_theta0(vid)
        theta_tn = theta_and_t_at_tn(vid, tn)[1]
        noms = [theta0.n, theta_tn.n]
        stds = [theta0.s, theta_tn.s]
        data_at_tn.append(unp.uarray(noms, stds))
    return np.array(data_at_tn)

# functie om plot en fit uit te voeren voor theta(tn) vs theta0 voor alle t in tn
# returnt een array met fit parameters voor elke fit, waaruit w1 berekend kan worden
def plot_and_fit_theta_vs_theta0(data):
    plot_theta_vs_theta0(data) 
    fit_params = fit_theta_vs_theta0(data) 
    return fit_params 

# functie om plot van theta(tn) vs theta0 te maken voor alle t in tn
def plot_theta_vs_theta0(data):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()

    n = 0
    for t in tn:
        data_at_t = data_tn(data, t)
        theta_0 = data_at_t[:,0]
        theta_t = data_at_t[:,1]
        plot_single_tn(ax, theta_0, theta_t, n)
        n += 1

    ax.set_xlabel(r'beginhoek $\theta_0$ (rad)')
    ax.set_ylabel(r'hoek $\theta(t_n)$ (rad)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_title(r'$\theta(t_n)$ (hoek op tijdstip $t_n$) tegen $\theta_0$ (beginhoek)')
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()

# functie om fit van theta(tn) vs theta0 uit te voeren voor alle t in tn
def fit_theta_vs_theta0(data):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()

    fit_params = []

    n = 0
    for t in tn:
        data_at_t = data_tn(data, t)
        theta_0 = data_at_t[:,0]
        theta_t = data_at_t[:,1]
        fit_params.append(fit_single_tn(ax, theta_0, theta_t, n))
        n += 1

    ax.set_xlabel(r'beginhoek $\theta_0$ (rad)')
    ax.set_ylabel(r'hoek $\theta(t_n)$ (rad)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_title(r'$\theta(t_n)$ (hoek op tijdstip $t_n$) tegen $\theta_0$ (beginhoek)')
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()
    return np.array(fit_params)

# functie om een plot voor een specifieke tn te maken
def plot_single_tn(ax, x, y, n):
    x_n = unp.nominal_values(x)
    y_n = unp.nominal_values(y)
    x_s = unp.std_devs(x)
    y_s = unp.std_devs(y)

    ax.errorbar(x_n, y_n, xerr=x_s, yerr=y_s, fmt='.', label=fr'$t_{{{n}}}={tn[n]}$')

# functie om een fit voor een specifieke tn uit te voeren, fit code gekopieerd van het bijgegeven fit script
def fit_single_tn(ax, x, y, n):
    x_n = unp.nominal_values(x)
    y_n = unp.nominal_values(y)
    x_s = unp.std_devs(x)
    y_s = unp.std_devs(y)

    ##### Gebruikersinput
    ### (2) Modelfunctie

    # Rechte lijn y = A + Bx
    # B is een vector met de parameters, in dit geval twee (A = B[0], B = B[1])
    # x is de array met de x waarden
    def f(B, x):
        return B[0] + B[1]*x

    # Uitgecomment: parabool y = A + Bx + C x^2
    # In dit geval heeft de vector B dus drie elementen (A = B[0], B = B[1], C = B[2])
    # def f(B, x):
    #     return B[0] + B[1]*x + B[2]*x**2

    ### (3) Startwaarden voor parameters
    B_start = [0.5, 2.0] # Voor de rechte lijn
    # B_start = [0.5, 2.0, 0.1] # Voor de parabool

    # (3a) Het is goed om de dataset te plotten samen met het model met de startwaarden 
    # ingevuld. Zo kun je inschatten of de startwaarden voor parameters goed zijn. 
    # ax.errorbar(x_n,y_n,xerr=x_s,yerr=y_s,fmt='o')
    # ax.plot(x_n, f(B_start,x_n), '-')

    ##### Fit

    ###  (4) Definieer het model-object om te gebruiken in odr
    odr_model = odr.Model(f)

    ### (5) Definieer een RealData object
    ## Een RealData-object vraagt om de onzekerheden in beide richtingen. 
    ## !! De onzekerheid in de x-richting mag ook nul zijn (dan mag je sx=0 weglaten), 
    ## maar dan moet bij onderdeel (6)/(6a) wel gekozen worden voor een
    ## kleinste-kwadratenaanpassing !!
    odr_data  = odr.RealData(x_n,y_n,sx=x_s,sy=y_s)

    ### (6) Maak een ODR object met data, model en startwaarden
    ## Je geeft startwaarden voor parameters mee bij keyword beta0
    odr_obj   = odr.ODR(odr_data,odr_model,beta0=B_start)
    ### (6a) Stel in op kleinste kwadraten (optioneel)
    ## Als de onzekerheden in de x-richting gelijk zijn aan nul, dan faalt de 
    ## default-aanpak van ODR. Je moet dan als methode een kleinste-kwadraten- 
    ## aanpassing instellen. Dit gaat met het volgende commando (nu uit):
    #odr_obj.set_job(fit_type=2)

    ### (7) Voer de fit uit
    ## Dit gebeurt expliciet door functie .run() aan te roepen
    odr_res   = odr_obj.run()

    ##### Output

    ### (8) Haal resultaten uit het resultaten-object:
    # (8a) De beste schatters voor de parameters
    par_best = odr_res.beta
    # (8b) De (EXTERNE!) onzekerheden voor deze parameters
    par_sig_ext = odr_res.sd_beta
    # (8c) De (INTERNE!) covariantiematrix
    par_cov = odr_res.cov_beta 
    #print(" De (INTERNE!) covariantiematrix  = \n", par_cov)
    # (8d) De chi-kwadraat en de gereduceerde chi-kwadraat van deze aanpassing
    chi2 = odr_res.sum_square
    #print("\n Chi-squared         = ", chi2)
    chi2red = odr_res.res_var
    #print(" Reduced chi-squared = ", chi2red, "\n")
    # (8e) Een compacte weergave van de belangrijkste resultaten als output
    #odr_res.pprint()

    ### (9) Plot van de fit met de dataset. 
    # De plot is onopgemaakt, je zult zelf asbijschriften moeten toevoegen, 
    # plotbereik kiezen etc. om hem mooi te maken. 
    # Om het model wat meer punten te geven, maken we daarvoor een aparte lijst x-waarden 
    xplot = np.linspace(np.min(x_n),np.max(x_n),num=100)
    # sig_xplot = np.linspace(np.min(x_s),np.max(x_s),num=100)
    # sig_yplot = np.linspace(np.min(y_s),np.max(y_s),num=100)
    line, = ax.plot(xplot,f(par_best,xplot),'-',label=fr'$t_{{{n}}}={tn[n]}$')
    ax.errorbar(x_n,y_n,xerr=x_s,yerr=y_s,fmt='.',color=line.get_color())
    # Naar keuze: exporteren als png of pdf in een directory naar keuze
    #plt.savefig('fit.png', dpi=300, bbox_inches='tight')
    #plt.savefig('fit.pdf', bbox_inches='tight')

    return np.array([par_best, par_sig_ext, par_cov, chi2, chi2red], dtype=object)

# functie om alle w1 schattingen te berekenen op basis van het theoretisch model
# helling wordt gehaald uit de fit parameters
def collect_w1s(fit_params):
    best_pars = fit_params[:,0]
    sig_pars = fit_params[:,1]
    w1_noms = []
    w1_stds = []
    for i in range(len(tn)):
        slope_val = best_pars[i][1]
        slope_unc = sig_pars[i][1]
        slope = unc.ufloat(slope_val, slope_unc)
        t = unc.ufloat(tn[i], convert_frame_to_time(frame_unc, fps))
        w1 = unp.arccos(slope)/t
        w1_noms.append(w1.n)
        w1_stds.append(w1.s)
    return unp.uarray(w1_noms, w1_stds)

# algemene functie om de gewogen beste schatter voor een dataset te krijgen
def get_best_weighted_guess(data):
    noms = unp.nominal_values(data)
    stds = unp.std_devs(data)
    weights = 1 / (stds**2)
    weighted_avg = np.sum(noms * weights) / np.sum(weights)
    weighted_std = np.sqrt(1 / np.sum(weights))
    w1 = unc.ufloat(weighted_avg, weighted_std)
    return w1

# algemene functie die een korte strijdigheidsanalyse uitvoert op basis van het 2 sigma criterium
def conflict_analysis(data, ref):
    delta = abs(ref.n - data.n)
    delta_sig = (ref.s**2 + data.s**2)**0.5
    conflict = delta > 2*delta_sig
    return conflict

# algemene functie die de gereduceerde chi2 berekent van data ten opzichte van een referentiewaarde
def red_chi2(data, ref):
    noms = unp.nominal_values(data)
    stds = unp.std_devs(data)
    chi2 = np.sum(((noms - ref.n)**2) / (stds**2 + ref.s**2))
    red_chi2 = chi2 / (len(data) - 1)
    return red_chi2

# algemene functie die checkt of een beste schatter binnen een bepaald percentage van een referentie ligt
def best_guess_within_percentage(best_guess, percentage, ref=w1_theorie):
    max = ref.n * (1 + percentage/100)
    min = ref.n * (1 - percentage/100)
    return (best_guess.n <= max) and (best_guess.n >= min)

# functie om de w1 metingen en beste schatter te plotten met een referentie waarde
def plot_w1(w1s, best_guess, ref=w1_theorie):
    w1_n = unp.nominal_values(w1s)
    w1_s = unp.std_devs(w1s)
    metingen = np.arange(len(w1s))
    plt.figure()
    plt.hlines(ref.n, xmin = -1, xmax = len(w1s)+1, color='r', label=r'$\omega_1$')
    plt.fill_between([-1, len(w1s)+1], ref.n - ref.s, ref.n + ref.s, color='r', alpha=0.2, label=r'$\omega_1 \pm \sigma$')
    plt.fill_between([-1, len(w1s)+1], ref.n + ref.s, ref.n + 2*ref.s, color='r', alpha=0.1, label=r'$\omega_1 \pm 2\sigma$')
    plt.fill_between([-1, len(w1s)+1], ref.n - ref.s, ref.n - 2*ref.s, color='r', alpha=0.1)
    plt.hlines(best_guess.n, xmin = -1, xmax = len(w1s)+1, color='b', label=r'$\overline{\omega}_{1}$')
    plt.fill_between([-1, len(w1s)+1], best_guess.n - best_guess.s, best_guess.n + best_guess.s, color='b', alpha=0.2, label=r'$\overline{\omega}_{1} \pm \hat{\sigma}$')
    plt.fill_between([-1, len(w1s)+1], best_guess.n + best_guess.s, best_guess.n + 2*best_guess.s, color='b', alpha=0.1, label=r'$\overline{\omega}_{1} \pm 2\hat{\sigma}$')
    plt.fill_between([-1, len(w1s)+1], best_guess.n - best_guess.s, best_guess.n - 2*best_guess.s, color='b', alpha=0.1)
    plt.errorbar(metingen, w1_n, yerr=w1_s, fmt='.', label=r'$\omega_1$ metingen')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.title(r'$\omega_1$ analyse')
    plt.xlabel('meting')
    plt.ylabel(r'$\omega_1$ (rad/s)')
    plt.xlim(0, len(w1s))
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()

# directory en output bestand instellen
loc = os.path.dirname(__file__)
os.chdir(loc)
fname = 'output.txt'

# De data regelrecht uit het databestand
raw_data = np.genfromtxt(fname, delimiter='\t')
# De data met toegevoegde onzekerheden
raw_data_unc = add_uncertainties_to_data(raw_data, frame_unc, pixel_pos_unc, pixel_pos_unc)
# De data omgezet naar fysieke eenheden (dus frame nummer -> seconde, pixel -> meter)
phys_data = convert_video_data_to_physical(raw_data_unc, pixel_scale, fps)
# Data met hoek data toegevoegd
angle_data = add_angle_data(phys_data, l)
# De data gescheiden per video
data_per_video = get_data_per_video(angle_data)
# Verzamel alle fit parameters voor elke t in tn, na uitvoeren van plot en fit
fit_params = plot_and_fit_theta_vs_theta0(data_per_video)
# Alle schattingen van w1 op basis van de fits
w1s = collect_w1s(fit_params)
# De beste schatter voor w1 op basis van gewogen gemiddelde
w1_best_guess = get_best_weighted_guess(w1s)
# Check of de beste schatter strijdig is met de theoretische waarde
strijdig = conflict_analysis(w1_best_guess, w1_theorie)
# Bereken de gereduceerde chi2 van de w1 metingen ten opzichte van de theoretische waarde
red_chi2_w1 = red_chi2(w1s, w1_theorie)
# Check of de beste schatter binnen 20% van de theoretische waarde ligt, zoals in de hypothese was verwacht
binnen_20_procent = best_guess_within_percentage(w1_best_guess, 20)
# Plot de verzamelde w1 schattingen met de beste schatter en de theoretische waarde
plot_w1(w1s, w1_best_guess)

# Print de informatie naar de console voor de rapportage en het labjournaal
print('--------------------------------------------------------')
print('Resultaten van de w1 meting en analyse:')
print(f'w1 theorie:                 {w1_theorie}')
print(f'w1 gemeten:                 {w1_best_guess}')
print(f'Strijdig:                   {strijdig}')
print(f'Red chi2:                   {red_chi2_w1}')
print(f'Binnen 20% van theorie:     {binnen_20_procent}')
print('--------------------------------------------------------')
