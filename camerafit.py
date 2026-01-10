# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:33:01 2017

@author: capel102
Versie 2024

Met dit script kun je een fit aan een set data maken. Fitten is: parameters in 
een model zo aanpassen dat de modelwaarden het best overeenkomen met de gemeten data.
De wiskundige achtergrond van fitten en de precieze werking van de fitroutine 
wordt behandeld in het onderdeel DATA-V.

Als gebruiker geef je als input:
    (1) De data. We nemen aan dat de dataset bestaat uit waarden voor de onafhankelijke 
    grootheid (in het script x genoemd) met onzekerheden (sig_x), en bijbehorende
    waarden voor de afhankelijke grootheid (deze noemen we y) met onzekerheden (sig_y)
    (2) Het model
    (3) Startwaarden voor de parameters, zodat de fit vlakbij de beste parameterwaardes start
    
(4)-(7) zijn de commando's waarmee de fit wordt gedefinieerd en uitgevoerd. 
    
Als output krijg je:
    (8) De beste schatters voor parameterwaarden met onzekerheden en overige 
    statistische informatie (dit wordt in de loop van DATA-V behandeld)
    (9) Een (onopgemaakte) plot met data en het model met de beste schatters voor 
    de parameters ingevuld. 

De standaardopzet van het script doet een fit volgens het model y = A + Bx, en in de 
comments hebben we een kwadratisch model voorgeprogrammeerd. Je kunt zelf een ander
model definiÃ«ren dat goed bij jouw situatie past.
"""

# Importeer de relevante packages
import scipy.odr as odr
import numpy as np
import matplotlib.pyplot as plt
import os

loc = os.path.dirname(__file__)
os.chdir(loc)
fname = 'output/slinger.dat'
print(fname)

data = np.genfromtxt(fname, delimiter='\t')

x = (512 - data[:,0])*0.05
sig_x = [0.021]*len(x)
y = 1/240 * np.arange(len(x))
sig_y = [1/240]*len(y)
print(np.max(data[:,0]))


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
plt.figure()
plt.errorbar(x,y*y,xerr=sig_x,yerr=sig_y,fmt='ko')
plt.plot(x, f(B_start,x), 'b-')
plt.show()

#%%

##### Fit

###  (4) Definieer het model-object om te gebruiken in odr
odr_model = odr.Model(f)

### (5) Definieer een RealData object
## Een RealData-object vraagt om de onzekerheden in beide richtingen. 
## !! De onzekerheid in de x-richting mag ook nul zijn (dan mag je sx=0 weglaten), 
## maar dan moet bij onderdeel (6)/(6a) wel gekozen worden voor een
## kleinste-kwadratenaanpassing !!
odr_data  = odr.RealData(x,y*y,sx=sig_x,sy=sig_y)

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

#%%

##### Output

### (8) Haal resultaten uit het resultaten-object:
# (8a) De beste schatters voor de parameters
par_best = odr_res.beta
# (8b) De (EXTERNE!) onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta
# (8c) De (INTERNE!) covariantiematrix
par_cov = odr_res.cov_beta 
print(" De (INTERNE!) covariantiematrix  = \n", par_cov)
# (8d) De chi-kwadraat en de gereduceerde chi-kwadraat van deze aanpassing
chi2 = odr_res.sum_square
print("\n Chi-squared         = ", chi2)
chi2red = odr_res.res_var
print(" Reduced chi-squared = ", chi2red, "\n")
# (8e) Een compacte weergave van de belangrijkste resultaten als output
odr_res.pprint()

### (9) Plot van de fit met de dataset. 
# De plot is onopgemaakt, je zult zelf asbijschriften moeten toevoegen, 
# plotbereik kiezen etc. om hem mooi te maken. 
# Om het model wat meer punten te geven, maken we daarvoor een aparte lijst x-waarden 
xplot = np.linspace(np.min(x),np.max(x),num=100)
# sig_xplot = np.linspace(np.min(sig_x),np.max(sig_x),num=100)
# sig_yplot = np.linspace(np.min(sig_y),np.max(sig_y),num=100)
plt.figure()
plt.title('Valtijd geplot tegen gevallen hoogte')
plt.xlabel('Gevallen hoogte h in cm')
plt.ylabel(r"Tijd $t^{2}$ in $s^{2}$")
plt.errorbar(x,y*y,xerr=sig_x,yerr=sig_y,fmt='ko')
plt.plot(xplot,f(par_best,xplot),'r-')
plt.show()
# Naar keuze: exporteren als png of pdf in een directory naar keuze
plt.savefig('fit.png', dpi=300, bbox_inches='tight')
#plt.savefig('fit.pdf', bbox_inches='tight')