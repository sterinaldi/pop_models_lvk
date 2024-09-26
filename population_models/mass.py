import numpy as np
from numba import njit
from math import erf
from figaro.utils import recursive_grid

@njit
def smoothing(m, mmin, delta):
    p            = np.zeros(m.shape, dtype = np.float64)
    idx          = (m > mmin) & (m < mmin + delta)
    shifted_mass = (m[idx] - mmin) / delta
    exponent     = 1. / shifted_mass - 1. / (1. - shifted_mass)
    p[idx]       = 1./(1.+np.exp(exponent))
    p[m >= mmin + delta] = 1.
    return p

@njit
def smoothing_float(m, mmin, delta):
    if m > mmin + delta:
        p = 1.
    elif m < mmin:
        p = 0.
    else:
        shifted_mass = (m - mmin) / delta
        exponent     = 1. / shifted_mass - 1. / (1. - shifted_mass)
        p            = 1./(1.+np.exp(exponent))
    return p

# Primary mass
@njit
def powerlaw_truncated(m, alpha, mmin, mmax):
    p = m**-alpha * (alpha-1.)/(mmin**(1.-alpha)-mmax**(1.-alpha))
    p[(m < mmin) | (m > mmax)] = 0.
    return p

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
    p[(m < mmin) | (m > mmax)] = 0.
    return p/(erf((mmax-mu)/(sigma*np.sqrt(2))) - erf((mmin-mu)/(sigma*np.sqrt(2))))*2.

@njit
def _peak_smoothed_unnorm(m, mu, sigma, mmin, delta):
    return peak(m, mu, sigma, mmin)*smoothing(m, mmin, delta)

@njit
def peak_smoothed(m, mu, sigma, mmin, delta, mmax = 100.):
    x  = np.linspace(mmin, mmax, 1000)
    dx = x[1]-x[0]
    n  = np.sum(_peak_smoothed_unnorm(x, mu, sigma, mmin, delta)*dx)
    return _peak_smoothed_unnorm(m.flatten(), mu, sigma, mmin, delta)/n

@njit
def plpeak(m, alpha, mmin, mmax, delta, mu, sigma, weight):
    return (1.-weight)*powerlaw_smoothed(m, alpha, mmax, mmin, delta) + weight*peak_smoothed(m, mu, sigma, mmin, delta)

# mass ratio
q_norm = np.linspace(0,1,101)[1:]
dq     = q_norm[1]-q_norm[0]

# Primary mass
@njit
def powerlaw_massratio_truncated(q, m1, beta, mmin):
    p = q**beta * (beta+1.) / (1. - (mmin/m1)**(beta+1))
    p[(m1 < mmin) | (q < mmin/m1)] = 0.
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
    return _powerlaw_massratio_unnorm(q, m1, beta, mmin, delta).flatten()

# LVK
@njit
def _plpeak_lvk_unnorm(m, alpha, mmin, mmax, delta, mu, sigma, weight):
    return ((1.-weight)*powerlaw_truncated(m, alpha, mmin, mmax) + weight*peak(m, mu, sigma, mmin))*smoothing(m, mmin, delta)

# LVK
@njit
def plpeak_lvk(m, alpha, mmin, mmax, delta, mu, sigma, weight):
    x  = np.linspace(mmin, mmax, 1000)
    dx = x[1]-x[0]
    n  = np.sum(_plpeak_lvk_unnorm(x, alpha, mmin, mmax, delta, mu, sigma, weight)*dx)
    return _plpeak_lvk_unnorm(m, alpha, mmin, mmax, delta, mu, sigma, weight)/n

from gwpopulation.models.mass import BaseSmoothedMassDistribution, two_component_single, power_law_mass
from gwpopulation.utils import truncnorm

def single_peak(mass, mmin, mpp, sigpp, gaussian_mass_maximum=100):
    return truncnorm(mass, mu=mpp, sigma=sigpp, high=gaussian_mass_maximum, low=mmin)

class PowerLaw(BaseSmoothedMassDistribution):
    primary_model = power_law_mass

class SinglePeak(BaseSmoothedMassDistribution):
    primary_model = single_peak
    
    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum = self.mmax)

class PLPeak(BaseSmoothedMassDistribution):
    primary_model = two_component_single
    
    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum = self.mmax)

pl_instance     = PowerLaw()
peak_instance   = SinglePeak()
plpeak_instance = PLPeak()

def plpeak_pl(m, q, alpha, mmin, mmax, delta, mu, sigma, weight, beta):
    """
    Wrapper for the model implemented in GWPopulation
    """
    pars_dict = {'alpha': alpha,
                 'mmin': mmin,
                 'mmax': mmax,
                 'delta_m': delta,
                 'mpp': mu,
                 'sigpp': sigma,
                 'lam': weight,
                 'beta': beta,
                 }
    dataset = {'mass_1': m, 'mass_ratio': q}
    return np.nan_to_num(plpeak_instance(dataset, **pars_dict), nan = -np.inf, posinf = -np.inf)

def pl_pl(m, q, alpha, mmin, mmax, delta, beta):
    pars_dict = {'alpha': alpha,
                 'mmin': mmin,
                 'mmax': mmax,
                 'delta_m': delta,
                 'beta': beta,
                 }
    dataset = {'mass_1': m, 'mass_ratio': q}
    return pl_instance(dataset, **pars_dict)

def peak_pl(m, q, mu, sigma, mmin, delta, beta):
    pars_dict = {'mmin': mmin,
                 'delta_m': delta,
                 'mpp': mu,
                 'sigpp': sigma,
                 'beta': beta,
                 }
    dataset = {'mass_1': m, 'mass_ratio': q}
    return peak_instance(dataset, **pars_dict)
