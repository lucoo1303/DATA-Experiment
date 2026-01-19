import math
import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
import warnings
import math
import os

warnings.filterwarnings("ignore")

# defining meassured constants with their uncertainties and tags for tracking purposes
m = unc.ufloat(92.28, 0.01, 'mass bob')*0.001 # mass of bob in kg
M = unc.ufloat(1459.03, 0.07, 'mass rod')*0.001 # mass of rod in kg
l = unc.ufloat(94.625, 0.025, 'length bob')*0.01 # length to center of mass of bob in m
L = unc.ufloat(102.05, 0.05, 'length rod')*0.01 # length of rod in m
h_spring = unc.ufloat(89.25, 0.05, 'spring height')*0.01 # length to spring attachment in m
m_spring = unc.ufloat(38.13, 0.45, 'mass on spring')*0.001 # mass hanging onto spring in kg
u = unc.ufloat(3.85, 0.11, 'displacement')*0.01 # displacement of mass hanging on spring under influence of gravity in m
g = unc.ufloat(9.80665, 0.00001, 'gravitational acceleration') # gravitational acceleration in m/s^2, with negligible uncertainty (but not zero, since that could cause issues in the uncertainty package)

# derived (constant) quantities with their uncertainties
I = m*l**2 + (M*L**2)/3
k = unc.ufloat((m_spring*g/u).n, (m_spring*g/u).s, 'spring constant')
# Created a new ufloat for k, since I won't track the error propagation of m_spring and u in the final experiment, 
# basically treating k as constant (with uncertainty) from here on out

# theoretical angular frequencies with their uncertainties
w1_theory = unp.sqrt((m*l*g + M*L*g/2)/I)
w2_theory = unp.sqrt((m*l*g + M*L*g/2 + 2*k*l*h_spring)/I)

# functions to calculate theoretical angles with uncertainties, based on time and initial conditions
def theta1_theory(t, theta01, theta02, w1, w2):
    return (theta01 + theta02)/2 * unp.cos(w1*t) + (theta01 - theta02)/2 * unp.cos(w2*t)

def theta2_theory(t, theta01, theta02, w1, w2):
    return (theta01 + theta02)/2 * unp.cos(w1*t) + (theta02 - theta01)/2 * unp.cos(w2*t)


print(f'w1 theoretical: {w1_theory} rad/s')
print(f'w2 theoretical: {w2_theory} rad/s')
