import numpy as np
from numba import njit
from figaro.cosmology import dVdz_approx_planck15 as approx

zmax   = 2.3 # LVK paper
z_norm = np.linspace(0,zmax,1000)
dz     = z_norm[1]-z_norm[0]

@njit
def powerlaw(z, k):
    p = (1+z)**k
    return p

@njit
def _unnorm_powerlaw_redshift(z, k):
    reg_const = (1+zmax)/approx(zmax)
    return powerlaw(z, k)*approx(z)/(1+z) * reg_const

@njit
def powerlaw_redshift(z, k):
    norm = np.sum(_unnorm_powerlaw_redshift(z_norm,k)*dz)
    return _unnorm_powerlaw_redshift(z, k)/norm
