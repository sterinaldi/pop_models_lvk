import numpy as np
from numba import njit
from math import erf
from figaro.utils import recursive_grid

@njit
def smoothing(m, mmin, delta):
    idx          = (m > mmin)
    shifted_mass = (m - mmin) / delta
    exponent     = 1. / shifted_mass - 1. / (1. - shifted_mass)
    p       = 1./(1.+np.exp(exponent))*idx
    p[m >= mmin + delta] = 1.
    p[m<mmin] = 1e-10
    return p

@njit
def smoothing_float(m, mmin, delta):
    if m > mmin + delta:
        p = 1.
    elif m < mmin:
        p = 1e-10
    else:
        shifted_mass = (m - mmin) / delta
        exponent     = 1. / shifted_mass - 1. / (1. - shifted_mass)
        p            = 1./(1.+np.exp(exponent))
    return p

# Primary mass
@njit
def powerlaw_truncated(m, alpha, mmin, mmax):
    p = np.where((m < mmin) | (m > mmax), 1e-10, m**-alpha * (alpha-1.)/(mmin**(1.-alpha)-mmax**(1.-alpha)))
    return p

@njit
def broken_powerlaw(m, alpha1, alpha2, mbreak, mmin, mmax):
    nn = (mmax*(mmax/mbreak)**(-alpha2) - mbreak)/(1-alpha2) + (mmax*(mmax/mbreak)**(-alpha1) - mbreak)/(1-alpha1)
#    p  = np.where((m < mmin) | (m > mmax), 1e-7, np.where(m < mbreak, (m/mbreak)**(-alpha1), (m/mbreak)**(-alpha2)))
    p  = np.where(m < mbreak, (m/mbreak)**(-alpha1), (m/mbreak)**(-alpha2))
    return p/nn

@njit
def _broken_powerlaw_smoothed_unnorm(m, alpha1, alpha2, mbreak, mmin, mmax, delta):
    return broken_powerlaw(m, alpha1, alpha2, mbreak, mmin, mmax)*smoothing(m, mmin, delta)


@njit
def broken_powerlaw_smoothed(m, alpha1, alpha2, mbreak, mmin, mmax, delta):
    x  = np.linspace(mmin, mmax, 1000)
    dx = x[1]-x[0]
    n  = np.sum(_broken_powerlaw_smoothed_unnorm(x, alpha1, alpha2, mbreak, mmin, mmax, delta)*dx)
    return _broken_powerlaw_smoothed_unnorm(m.flatten(), alpha1, alpha2, mbreak, mmin, mmax, delta)/n

@njit
def _powerlaw_smoothed_unnorm(m, alpha, mmax, mmin, delta):
    return powerlaw_truncated(m, alpha, mmin, mmax)*smoothing(m, mmin, delta)
    
@njit
def powerlaw_smoothed(m, alpha, mmax, mmin, delta):
    x  = np.linspace(mmin, mmax, 1000)
    dx = x[1]-x[0]
    n  = np.sum(_powerlaw_smoothed_unnorm(x, alpha, mmax, mmin, delta)*dx)
    return _powerlaw_smoothed_unnorm(m.flatten(), alpha, mmax, mmin, delta)/n

@njit
def peak(m, mu, sigma, mmin, mmax = 100.):
    p = np.exp(-0.5*(m-mu)**2/sigma**2)/(np.sqrt(2*np.pi)*sigma)
    p[(m < mmin) | (m > mmax)] = 1e-20
    return p/(erf((mmax-mu)/(sigma*np.sqrt(2))) - erf((mmin-mu)/(sigma*np.sqrt(2))))*2.

@njit
def _peak_smoothed_unnorm(m, mu, sigma, mmin, delta, mmax = 100.):
    return peak(m, mu, sigma, mmin, mmax)*smoothing(m, mmin, delta)

@njit
def peak_smoothed(m, mu, sigma, mmin, delta, mmax = 100.):
    x  = np.linspace(mmin, mmax, 1000)
    dx = x[1]-x[0]
    n  = np.sum(_peak_smoothed_unnorm(x, mu, sigma, mmin, delta, mmax)*dx)
    return _peak_smoothed_unnorm(m.flatten(), mu, sigma, mmin, delta, mmax)/n

@njit
def plpeak(m, alpha, mmin, mmax, delta, mu, sigma, weight):
    return (1.-weight)*powerlaw_smoothed(m, alpha, mmax, mmin, delta) + weight*peak_smoothed(m, mu, sigma, mmin, delta, mmax)

# mass ratio
q_norm = np.linspace(0,1,1001)[1:]
dq     = q_norm[1]-q_norm[0]

# Primary mass
@njit
def powerlaw_massratio_truncated(q, m1, beta, mmin):
    p = np.where((m1 < mmin) | (q < mmin/m1), 0, q**beta * (beta+1.) / (1. - (mmin/m1)**(beta+1)))
    return p

@njit
def _powerlaw_massratio_for_normalisation(q, m1, beta, mmin, delta):
    return powerlaw_massratio_truncated(q, m1, beta, mmin)*smoothing(m1*q, mmin, delta)

@njit
def _powerlaw_massratio_unnorm(q, m1, beta, mmin, delta):
    return powerlaw_massratio_truncated(q, m1, beta, mmin)*smoothing(m1*q, mmin, delta)

@njit
def _powerlaw_massratio(q, m1, beta, mmin, delta):
    norm = np.sum(_powerlaw_massratio_for_normalisation(q_norm, m1, beta, mmin, delta)*dq)
    return _powerlaw_massratio_unnorm(q, m1, beta, mmin, delta)/norm

@njit
def powerlaw_massratio(q, m1, beta, mmin, delta):
    return _powerlaw_massratio(q, m1, beta, mmin, delta).flatten()

# LVK
@njit
def _plpeak_lvk_unnorm(m, alpha, mmin, mmax, delta, mu, sigma, weight):
    return ((1.-weight)*powerlaw_truncated(m, alpha, mmin, mmax) + weight*peak(m, mu, sigma, mmin, mmax))*smoothing(m, mmin, delta)

@njit
def _plpeak_lvk_np2p(m, alpha, mmin, mmax, delta, mu, sigma, log10_w):
    return ((1.- (10**log10_w))*powerlaw_truncated(m, alpha, mmin, mmax) + (10**log10_w)*peak(m, mu, sigma, mmin, 100.))*smoothing(m, mmin, delta)

# LVK
@njit
def plpeak_lvk(m, alpha, mmin, mmax, delta, mu, sigma, weight):
    x  = np.linspace(mmin, mmax, 1000)
    dx = x[1]-x[0]
    n  = np.sum(_plpeak_lvk_unnorm(x, alpha, mmin, mmax, delta, mu, sigma, weight)*dx)
    return _plpeak_lvk_unnorm(m, alpha, mmin, mmax, delta, mu, sigma, weight)/n

# Joint mass and mass ratio
grid, dgrid = recursive_grid([[1.001,100],[0.0101,1]], [100,100])
dgrid = np.prod(dgrid)
grid_m = np.copy(grid[:,0]).flatten()
grid_q = np.copy(grid[:,1]).flatten()

@njit
def plpeak_pl(m, q, alpha, mmin, mmax, delta, mu, sigma, weight, beta):
    p_norm = _plpeak_lvk_unnorm(grid_m, alpha, mmin, mmax, delta, mu, sigma, weight)*_powerlaw_massratio_unnorm(grid_q, grid_m, beta, mmin, delta)
    norm = np.sum(p_norm)*dgrid
    return _plpeak_lvk_unnorm(m, alpha, mmin, mmax, delta, mu, sigma, weight)*_powerlaw_massratio_unnorm(q, m, beta, mmin, delta)/norm

@njit
def pl_pl(m, q, alpha, mmin, mmax, delta, beta):
    p_norm = _powerlaw_smoothed_unnorm(grid_m, alpha, mmax, mmin, delta)*_powerlaw_massratio_unnorm(grid_q, grid_m, beta, mmin, delta)
    norm = np.sum(p_norm)*dgrid
    return _powerlaw_smoothed_unnorm(m, alpha, mmax, mmin, delta)*_powerlaw_massratio_unnorm(q, m, beta, mmin, delta)/norm

@njit
def peak_pl(m, q, mu, sigma, mmin, delta, beta):
    p_norm = _peak_smoothed_unnorm(grid_m, mu, sigma, mmin, delta)*_powerlaw_massratio_unnorm(grid_q, grid_m, beta, mmin, delta)
    norm = np.sum(p_norm)*dgrid
    return _peak_smoothed_unnorm(m, mu, sigma, mmin, delta)*_powerlaw_massratio_unnorm(q, m, beta, mmin, delta)/norm
