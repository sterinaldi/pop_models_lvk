import numpy as np
from numba import njit

zmax = 2.3 # LVK paper

@njit
def powerlaw(z, k):
    p = ((k+1.)*(1.+z)**k)/((1.+zmax)**(k+1.)-1.)
    p[z > zmax] = 0.
    return p
