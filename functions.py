import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp
import scipy.odr as odr
import matplotlib.pyplot as plt
import os


temp_var = 1
temp_unc = 0.1


# defining meassured constants with their uncertainties and tags for tracking purposes
m = unc.ufloat(temp_var, temp_unc, 'mass bob')
M = unc.ufloat(temp_var, temp_unc, 'mass rod')
l = unc.ufloat(temp_var, temp_unc, 'length bob')
L = unc.ufloat(temp_var, temp_unc, 'length rod')
h = unc.ufloat(temp_var, temp_unc, 'spring height')
g = unc.ufloat(temp_var, temp_unc, 'gravitational acceleration')
m_spring = unc.ufloat(temp_var, temp_unc, 'mass on spring')
u = unc.ufloat(temp_var, temp_unc, 'displacement')


# derived (constant) quantities with their uncertainties
I = m*l**2 + (M*L**2)/3
k = unc.ufloat(m_spring.n*g.n/u.n, m_spring.s*g.s/u.s, 'spring constant')
# Created a new ufloat for k, since I won't track the error propagation of m_spring and u in the final experiment, 
# basically treating k as constant (with uncertainty) from here on out

# theoretical angular frequencies with their uncertainties
w1_theory = unp.sqrt((m*l*g + M*L*g/2)/I)
w2_theory = unp.sqrt((m*l*g + M*L*g/2 + 2*k*l*h)/I)

# functions to calculate theoretical angles with uncertainties, based on time and initial conditions
def theta1_theory(t, theta01, theta02, w1, w2):
    return (theta01 + theta02)/2 * unp.cos(w1*t) + (theta01 - theta02)/2 * unp.cos(w2*t)

def theta2_theory(t, theta01, theta02, w1, w2):
    return (theta01 + theta02)/2 * unp.cos(w1*t) + (theta02 - theta01)/2 * unp.cos(w2*t)


print(f'{k.tag} = {k}')

loc = os.path.dirname(__file__)
os.chdir(loc)

# Locatie video
directory = 'videos/'
vidname='test.MOV'
video = directory + vidname
print(f'Video location: {video}')

