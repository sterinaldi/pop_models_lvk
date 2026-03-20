import numpy as np
from math import erf
from numba import njit

nchi = np.linspace(-1,1,1000)
dnchi = nchi[1]-nchi[0]

@njit
def truncated_gaussian(m, mu, sigma, mmin, mmax):
    p = np.exp(-0.5*(m-mu)**2/sigma**2)/(np.sqrt(2*np.pi)*sigma)
    p[(m < mmin) | (m > mmax)] = 1e-20
    return p/(erf((mmax-mu)/(sigma*np.sqrt(2))) - erf((mmin-mu)/(sigma*np.sqrt(2))))*2.

@njit
def joint_spin_magnitudes(a1, a2, mu, sigma):
    return truncated_gaussian(a1, mu, sigma, 0., 1.)*truncated_gaussian(a2, mu, sigma, 0., 1.)

@njit
def spin_magnitude(a, mu, sigma):
    return truncated_gaussian(a, mu, sigma, 0., 1.)

@njit
def joint_spin_tilt_angles(cost1, cost2, mu, sigma, zeta):
    return zeta*truncated_gaussian(cost1, mu, sigma, -1., 1.)*truncated_gaussian(cost2, mu, sigma, -1, 1.) + (1-zeta)/4.

@njit
def joint_spin_tilt_angles(cost, mu, sigma, zeta):
    return zeta*truncated_gaussian(cost, mu, sigma, -1., 1.) + (1-zeta)/2.

@njit
def _skewed_gaussian_chieff_unnorm(chieff, mu, sigma, epsilon):
    joint_up   = truncated_gaussian(np.zeros(1), mu, sigma*(1-epsilon), -1., 1.)*(1-epsilon)
    joint_down = truncated_gaussian(np.zeros(1), mu, sigma*(1+epsilon), -1., 1.)*(1+epsilon)
    p_up       = truncated_gaussian(chieff, mu, sigma*(1-epsilon), -1., 1.)*(1-epsilon)/joint_up
    p_down     = truncated_gaussian(chieff, mu, sigma*(1+epsilon), -1., 1.)*(1+epsilon)/joint_down
    return np.where(chieff < 0, p_down, p_up)

@njit
def skewed_gaussian_chieff(chieff, mu, sigma, epsilon):
    N = _skewed_gaussian_chieff_unnorm(nchi, mu, sigma, epsilon).sum()*dnchi
    return _skewed_gaussian_chieff_unnorm(chieff, mu, sigma, epsilon)/N

@njit
def gaussian_chip(chip, mu, sigma):
    return truncated_gaussian(chip, mu, sigma, -1., 1.)
