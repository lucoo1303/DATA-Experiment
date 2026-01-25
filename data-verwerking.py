import scipy.odr as odr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import uncertainties as unc
from uncertainties import unumpy as unp
import os
import warnings

# irritantie warnings over deprecated modules in de output console uitzetten
warnings.filterwarnings("ignore")


# ============================================
# CONSTANTEN
# ============================================

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

# Camera eigenschappen:
fps = 240
pixel_width = 512 # in pixels
pixel_pos_unc = 1.0 # onzekerheid in pixel positie 
frame_unc = 0.5 # onzekerheid in frame nummer 
frame_width = unc.ufloat(68.2, 0.1, 'frame breedte')*0.01 # in m
pixel_scale = frame_width / pixel_width # m per pixel


# tijdstippen waarop ik de data wil evalueren 
tn = np.round(np.arange(1, 8, 0.35), 3)
# geef elke tn een unieke kleur voor in de plot
colors = cm.tab20.colors 

# ============================================
# Functies voor data verkrijgen en omzetten
# ============================================

# Functie zet frame nummer om naar tijd op basis van fps
def convert_frame_to_time(frame_no, fps): return frame_no / fps

# Functie zet pixelpositie om naar fysieke positie op basis van schaal
def convert_pixel_to_physical(pixel_value, pixel_scale): return pixel_value * pixel_scale

# Functie om beginhoek uit data te halen
def get_theta0(data): return data[:,3][0]

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

# Functie om hoek data toe te voegen aan dataset
def add_angle_column(data, length):
    angle = unp.arcsin(data[:,1] / length)
    return np.column_stack((data, angle))


# ============================================
# Theta functies
# ============================================


# functie om benaderings onzekerheid in hoek te schatten
def uncertaninty_in_angle_due_to_approximation(angles, index):
    angle = angles[index]
    prev_angle = angles[index - 1].n
    next_angle = angles[index + 1].n if index + 1 < len(angles) else angles[index].n
    diff1 = np.abs(angle.n - prev_angle)/2
    diff2 = np.abs(angle.n - next_angle)/2
    angle_unc = diff1 + diff2 + angle.s
    return angle_unc

# functie om van een video hoek op een specifiek tijdstip te krijgen, geeft de hoek terug met bijgekomen benaderings onzekerheid
def theta_at_time(video, target_time):
    angles = video[:,3]
    i = np.argmin(np.abs(unp.nominal_values(video[:,0])-target_time))
    angle_n = angles[i].n
    angle_s = uncertaninty_in_angle_due_to_approximation(angles, i)
    return unc.ufloat(angle_n, angle_s)

# functie om van alle video's de data op een tijdstip t te verzamelen
# returnt een array met rijen voor elke video en kolommen voor theta0 en theta(t)
# als het ware returnt het dus een array met coordinaten voor een scatter plot van theta(t) vs theta0
def get_theta_vs_theta0(videos, t):
    theta_vs_theta0 = []
    for video in videos:
        theta0 = get_theta0(video)
        theta_t = theta_at_time(video, t)
        noms = [theta0.n, theta_t.n]
        stds = [theta0.s, theta_t.s]
        theta_vs_theta0.append(unp.uarray(noms, stds))
    return np.array(theta_vs_theta0)



# ============================================
# Plotten en fitten
# ============================================


# functie om plot en fit uit te voeren voor theta(tn) vs theta0 voor alle t in tn
# returnt een array met fit parameters voor elke fit, waaruit w1 gehaald kan worden
def plot_and_fit_theta_vs_theta0(data):
    plot_theta_vs_theta0(data) 
    fit_params = fit_theta_vs_theta0(data) 
    return fit_params 

# functie om plot van theta(tn) vs theta0 te maken voor alle t in tn
def plot_theta_vs_theta0(data):
    fig = plt.figure(figsize=(10, 5))
    
    for n, t in enumerate(tn):
        data_at_t = get_theta_vs_theta0(data, t)
        theta_0 = data_at_t[:,0]
        theta_t = data_at_t[:,1]
        x_n = unp.nominal_values(theta_0)
        y_n = unp.nominal_values(theta_t)
        x_s = unp.std_devs(theta_0)
        y_s = unp.std_devs(theta_t)
        plt.errorbar(x_n, y_n, xerr=x_s, yerr=y_s, fmt='.', label=fr'$t_{{{n}}}={tn[n]}$', color=colors[n])

    plt.xlabel(r'beginhoek $\theta_0$ (rad)')
    plt.ylabel(r'hoek $\theta(t_n)$ (rad)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.title(r'$\theta(t_n)$ (hoek op tijdstip $t_n$) tegen $\theta_0$ (beginhoek)')
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig('Plots/theta_vs_theta0_all_tn.png', dpi=300)
    plt.show()

# functie om fit van theta(tn) vs theta0 uit te voeren voor alle t in tn
def fit_theta_vs_theta0(data):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()

    fit_params = []

    for n, t in enumerate(tn):
        data_at_t = get_theta_vs_theta0(data, t)
        theta_0 = data_at_t[:,0]
        theta_t = data_at_t[:,1]
        fit_params.append(execute_fit(ax, theta_0, theta_t, n))

    ax.set_xlabel(r'beginhoek $\theta_0$ (rad)')
    ax.set_ylabel(r'hoek $\theta(t_n)$ (rad)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_title(r'$\theta(t_n)$ (hoek op tijdstip $t_n$) tegen $\theta_0$ (beginhoek)')
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig('Plots/theta_vs_theta0_all_tn_fit.png', dpi=300)
    plt.show()
    return np.array(fit_params)

# functie om een fit voor een specifieke tn uit te voeren, fit code gekopieerd van het bijgegeven fit script
def execute_fit(ax, x, y, n):
    # de nominale waardes en onzekerheden van x en y
    x_n = unp.nominal_values(x)
    y_n = unp.nominal_values(y)
    x_s = unp.std_devs(x)
    y_s = unp.std_devs(y)

    # het model definiëren
    def f(B, x):
        return x * np.cos(B[0] * tn[n])
    
    # startwaarden instellen en ODR object maken
    B_start = [w1_theorie.n] 
    odr_model = odr.Model(f)
    odr_data  = odr.RealData(x_n,y_n,sx=x_s,sy=y_s)
    odr_obj   = odr.ODR(odr_data,odr_model,beta0=B_start)
    odr_res   = odr_obj.run()

    # De resultaten van de fit 
    par_best = odr_res.beta
    par_sig_ext = odr_res.sd_beta
    par_cov = odr_res.cov_beta 
    chi2 = odr_res.sum_square
    chi2red = odr_res.res_var

    xplot = np.linspace(np.min(x_n),np.max(x_n),num=100)
    ax.plot(xplot,f(par_best,xplot),'-',label=fr'$t_{{{n}}}={tn[n]}$', color=colors[n])
    ax.errorbar(x_n,y_n,xerr=x_s,yerr=y_s,fmt='.',color=colors[n])

    return np.array([par_best, par_sig_ext, par_cov, chi2, chi2red], dtype=object)

# functie om de individuele w1 bepalingen en beste schatter te plotten met een referentie waarde
def plot_all_w1s(w1s, best_guess, show_outliers=False, outliers=[], outlier_indices=[], ref=w1_theorie):
    w1_n = unp.nominal_values(w1s)
    w1_s = unp.std_devs(w1s)
    metingen = np.arange(len(w1s))
    outliers_n = unp.nominal_values(outliers)
    outliers_s = unp.std_devs(outliers)
    plt.figure()
    plt.hlines(ref.n, xmin = -1, xmax = len(w1s)+1, color='r', label=r'$\omega_1$ (theorie)')
    plt.fill_between([-1, len(w1s)+1], ref.n - ref.s, ref.n + ref.s, color='r', alpha=0.2, label=r'$\omega_1 \pm \sigma$')
    plt.fill_between([-1, len(w1s)+1], ref.n + ref.s, ref.n + 2*ref.s, color='r', alpha=0.1, label=r'$\omega_1 \pm 2\sigma$')
    plt.fill_between([-1, len(w1s)+1], ref.n - ref.s, ref.n - 2*ref.s, color='r', alpha=0.1)
    plt.hlines(best_guess.n, xmin = -1, xmax = len(w1s)+1, color='b', label=r'$\overline{\omega}_{1}$ (beste schatter)')
    plt.fill_between([-1, len(w1s)+1], best_guess.n - best_guess.s, best_guess.n + best_guess.s, color='b', alpha=0.2, label=r'$\overline{\omega}_{1} \pm \hat{\sigma}$')
    plt.fill_between([-1, len(w1s)+1], best_guess.n + best_guess.s, best_guess.n + 2*best_guess.s, color='b', alpha=0.1, label=r'$\overline{\omega}_{1} \pm 2\hat{\sigma}$')
    plt.fill_between([-1, len(w1s)+1], best_guess.n - best_guess.s, best_guess.n - 2*best_guess.s, color='b', alpha=0.1)
    plt.errorbar(metingen, w1_n, yerr=w1_s, fmt='.', label=r'$\omega_1$ metingen')
    if show_outliers:
        plt.errorbar(outlier_indices, outliers_n, yerr=outliers_s, fmt='.', color='orange', label='Uitschieters')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.xlabel(r'$n$ (index $t_n$)')
    plt.ylabel(r'$\omega_1$ (rad/s)')
    plt.xlim(-0.5, len(w1s)-0.5)
    plt.xticks(metingen[metingen % 2 == 0])
    if show_outliers:
        plt.title(r'$\omega_1$ metingen met uitschieters')
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig('Plots/w1_all_measurements_with_outliers.png', dpi=300)
    elif not show_outliers and len(outliers) > 0:
        plt.title(r'Definitieve $\omega_1$ metingen')
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig('Plots/w1_correct_measurements.png', dpi=300)
    else:
        plt.title(r'Alle $\omega_1$ metingen')
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig('Plots/w1_all_measurements.png', dpi=300)
    plt.show()


# ============================================
# Data analyse functies
# ============================================

# functie om alle w1 schattingen te verzamelen uit de fit parameters
def collect_w1s(fit_params):
    w1_noms = fit_params[:,0]
    w1_stds = fit_params[:,1]
    return unp.uarray(np.abs(w1_noms), w1_stds)

# algemene functie om de gewogen beste schatter voor een dataset te krijgen
def get_best_weighted_guess(data):
    noms = unp.nominal_values(data)
    stds = unp.std_devs(data)
    weights = 1 / (stds**2)
    weighted_avg = np.sum(noms * weights) / np.sum(weights)
    weighted_std = np.sqrt(1 / np.sum(weights))
    w1 = unc.ufloat(weighted_avg, weighted_std)
    return w1

# functie om uitschieters binnen de w1 schattingen te vinden en te verwijderen uit de dataset op basis van de MAD methode
def identify_outliers(w1s, w1_best_guess):
    w1s_corrected = w1s.copy()
    outliers = []
    outlier_indices = []
    median_nom = np.median(unp.nominal_values(w1s))
    mad = np.median(np.abs(unp.nominal_values(w1s) - median_nom))
    factor = 4 # factor waarmee de MAD vermenigvuldigd wordt om de drempel te bepalen
    plot_all_w1s(w1s_corrected, w1_best_guess)
    for i, w1 in enumerate(w1s):
        if abs(unp.nominal_values(w1) - median_nom) > factor * mad:
            outliers.append(w1)
            outlier_indices.append(i)
            w1s_corrected[i] = None
    w1_best_guess = get_best_weighted_guess([w1 for w1 in w1s_corrected if w1 is not None])
    plot_all_w1s(w1s_corrected, w1_best_guess, True if len(outliers) > 0 else False, outliers, outlier_indices)
    return w1s_corrected, w1_best_guess, (outliers, outlier_indices)

# algemene functie die een korte strijdigheidsanalyse uitvoert op basis van het 2 sigma criterium
def conflict_analysis(best_guess, ref):
    delta = abs(ref.n - best_guess.n)
    delta_sig = (ref.s**2 + best_guess.s**2)**0.5
    conflict = delta > 2*delta_sig
    return conflict

# algemene functie die de gereduceerde chi2 berekent van data ten opzichte van een referentiewaarde
def calc_red_chi2(data, ref, degrees_of_freedom=0):
    data = np.array([d for d in data if d is not None])
    noms = unp.nominal_values(data)
    stds = unp.std_devs(data)
    chi2 = np.sum(((noms - ref.n)**2) / (stds**2 + ref.s**2))
    red_chi2 = chi2 / (len(data) - degrees_of_freedom)
    return red_chi2

# algemene functie die checkt of een beste waarde binnen een bepaald percentage van een referentie ligt
def is_val_within_percentage_of_ref(best_guess, percentage, ref=w1_theorie):
    max = ref.n * (1 + percentage/100)
    min = ref.n * (1 - percentage/100)
    return (best_guess.n <= max) and (best_guess.n >= min)


# ============================================
# Extra theorie vs experiment vergelijking
# ============================================


def theta_model(t, theta0, w1):
    return theta0 * unp.cos(w1*t)

def plot_theory_vs_experiment(w1_theory, w1_exp, theta0):
    times = np.linspace(0, 8, 1000)
    theta_theory = theta_model(times, theta0, w1_theory)
    theta_exp = theta_model(times, theta0, w1_exp)

    plt.figure(figsize=(10,5))
    plt.plot(times, unp.nominal_values(theta_theory), label=r'Tijdsverloop met theoretische $\omega_1$')
    plt.plot(times, unp.nominal_values(theta_exp), label=r'Tijdsverloop met experimentele $\omega_1$')
    plt.xlabel('Tijd (s)')
    plt.ylabel('Hoek (rad)')
    plt.title('Theoretische en Experimentele Hoek over Tijd')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig('Plots/theory_vs_experiment.png', dpi=300)
    plt.show()


# ============================================
# Hoofdscript
# ============================================

# directory en output bestand instellen
loc = os.path.dirname(__file__)
os.chdir(loc)
dir_write = 'Data/'
filename_data = 'output.txt'
fname = dir_write + filename_data

# De data regelrecht uit het databestand
raw_data = np.genfromtxt(fname, delimiter='\t')
# De data met toegevoegde onzekerheden
raw_data_unc = add_uncertainties_to_data(raw_data, frame_unc, pixel_pos_unc, pixel_pos_unc)
# De data omgezet naar fysieke eenheden (dus frame nummer -> seconde, pixel -> meter)
phys_data = convert_video_data_to_physical(raw_data_unc, pixel_scale, fps)
# Data met hoek data toegevoegd
angle_data = add_angle_column(phys_data, l)
# De data gescheiden per video
data_per_video = get_data_per_video(angle_data)
# Verzamel alle fit parameters voor elke t in tn, na uitvoeren van plot en fit
fit_params = plot_and_fit_theta_vs_theta0(data_per_video)
# Alle schattingen van w1 op basis van de fits
w1s = collect_w1s(fit_params)
# De beste schatter voor w1 op basis van gewogen gemiddelde
w1_best_guess = get_best_weighted_guess(w1s)
# Check voor uitschieters in de w1 schattingen
w1s, w1_best_guess, outliers = identify_outliers(w1s, w1_best_guess)
# Check of de beste schatter strijdig is met de theoretische waarde
strijdig = conflict_analysis(w1_best_guess, w1_theorie)
# Bereken de gereduceerde chi2 van de w1 metingen ten opzichte van de theoretische waarde
red_chi2_w1 = calc_red_chi2(w1s, w1_theorie)
# Check of de beste schatter binnen 20% van de theoretische waarde ligt, zoals in de hypothese was verwacht
binnen_20_procent = is_val_within_percentage_of_ref(w1_best_guess, 20)

# Print de informatie naar de console voor de rapportage en het labjournaal
print('--------------------------------------------------------')
print('Resultaten van de w1 metingen:')
print(f'w1 metingen:                {[f"{w1:.2u}" if w1 is not None else "uitschieter" for w1 in w1s]}')
print(f'Uitschieters:               {[f"{w1:.2u}" for w1 in outliers[0]]} op n uit t_n is: {outliers[1]}')
print('--------------------------------------------------------')
print('Resultaten van de w1 analyse:')
print(f'w1 theorie:                 {w1_theorie}')
print(f'w1 beste schatter:          {w1_best_guess}')
print(f'Strijdig:                   {strijdig}')
print(f'Red chi2:                   {red_chi2_w1}')
print(f'Binnen 20% van theorie:     {binnen_20_procent}')
print('--------------------------------------------------------')

# Plot alle w1 schattingen met de beste schatter en de theoretische waarde, zonder uitschieters
plot_all_w1s(w1s, w1_best_guess, show_outliers=False, outliers=outliers[0], outlier_indices=outliers[1])

# Extra plot van theorie vs experiment voor een specifieke beginhoek
plot_theory_vs_experiment(w1_theorie, w1_best_guess, unc.ufloat(0.2, 0.01))