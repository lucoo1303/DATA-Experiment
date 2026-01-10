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
g = unc.ufloat(temp_var, temp_unc, 'gravitational acceleration')
m_spring = unc.ufloat(temp_var, temp_unc, 'mass on spring')
u = unc.ufloat(temp_var, temp_unc, 'displacement')


# derived (constant) quantities with their uncertainties and tags
k = m_spring/u
k.tag = 'spring constant'
I = m*l**2 + (M*L**2)/3
I.tag = 'moment of inertia'

# theoretical angular frequencies with their uncertainties and tags
w1_theory = unp.sqrt((m*l*g + M*L*g/2)/I)
w1_theory.tag = 'w1 theory'
w2_theory = unp.sqrt((m*l*g + M*L*g/2 + 2*k*l**2)/I)
w2_theory.tag = 'w2 theory'


# functions to calculate theoretical angles with uncertainties, based on time and initial conditions
def theta1_theory(t, theta01, theta02, w1, w2):
    theta1 = (theta01 + theta02)/2 * unp.cos(w1*t) + (theta01 - theta02)/2 * unp.cos(w2*t)
    theta1.tag = 'theta1 theory'
    return theta1

def theta2_theory(t, theta01, theta02, w1, w2):
    theta2 = (theta01 + theta02)/2 * unp.cos(w1*t) + (theta02 - theta01)/2 * unp.cos(w2*t)
    theta2.tag = 'theta2 theory'
    return theta2