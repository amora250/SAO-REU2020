import numpy as np
import os, sys
import matplotlib.pylab as plt
import pandas as pd
import scipy.interpolate
import emcee
import corner
import matplotlib as mpl

plt.style.use(['default','seaborn-bright','seaborn-ticks'])
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


def schechter_logL(logL, alpha, logLstar, logPhistar):
    """
    Schechter function LF for log L
    
    phi(log L) = phistar * 1n 10 * L * (L/Lstar)^alpha e^(-L/Lstar) / Lstar
               = phistar * 1n 10 * (L/Lstar)^(alpha+1) e^(-L/Lstar)
    """
    L       = 10**logL
    Lstar   = 10**logLstar
    Phistar = 10**logPhistar
    
    Phi_logL = np.log(10.) * Phistar * (L/Lstar)**(alpha+1) * np.exp(-L/Lstar)
    return Phi_logL


def ln_schechter_logL(logL, alpha, logLstar, logPhistar):
    """
    ln Schechter function LF (nicer for computing the likelihood)
    
    ln(phi(log L)) = ln(phistar) + ln(ln10) + (alpha+1)*ln(L/Lstar) -L/Lstar
    """    
    L          = 10**logL
    Lstar      = 10**logLstar
    Phistar    = 10**logPhistar
    LoverLstar = np.array(L/Lstar).astype('float64')
    
    ln_Phi_logL = np.log(np.log(10.)) + np.log(Phistar) + (alpha+1)*np.log(LoverLstar) - LoverLstar
    return ln_Phi_logL


def ln_likelihood(theta, logL, ndens):
    """
    log likelihood = -0.5 \Sum (data - model)^2
    
    use log data and model to make the likelihood a sensible number
    """
    lnlike = -0.5*np.nansum((np.log(ndens) - ln_schechter_logL(logL, *theta))**2.)
    return lnlike
        

def ln_prior(theta):
    """
    uniform prior on log Schechter parameters
    """
    alpha, logLstar, logPhistar = theta
    if -3 < alpha < 0 and 40. < logLstar < 44. and -6 < logPhistar < -2:
        return 0.
    else:
        return -np.inf


def ln_posterior(theta, logL, ndens):
    if ln_prior(theta) == 0:
        return ln_likelihood(theta, logL, ndens)
    else:
        return -np.inf

# --------------------------------------------------------
# Fitting functions

def fit_schechter_scipy(logL, LF_logL, 
                        logL_min=42.5, logL_max=43.5,
                        p0=[-1., 43, -3.5]):
    """
    Use scipy curve_fit to minimise chi2 = sum[ln(LF) - ln(LF_schechter)]
    
    Outputs:
        p = alpha, logLstar, logPhistar
    """
    
    inrange = np.where((logL >= logL_min) & (logL <= logL_max))[0]

    p, covar = scipy.optimize.curve_fit(ln_schechter_logL, logL[inrange], np.log(LF_logL[inrange]), 
                                        p0=p0, bounds=([-3, 40, -6], [0, 44, -2]),
                                        method='dogbox')
    error = np.zeros(len(p))
    for i in range(len(p)):
        error[i] = np.abs(np.sqrt(covar[i][i]))
        
    return p, error


def fit_schechter_emcee(logL, LF_logL, 
                        logL_min=42.5, logL_max=43.5,
                        p0=[-2., 43, -3.5],
                        nwalkers=200, nsteps=1000):
    """
    Find best fit Schechter function using MCMC
    
    Outputs:
        sampler    emcee sampler object
        
    """
    
    # run scipy to get starting position
    p, cov = fit_schechter_scipy(logL, LF_logL, logL_min=logL_min, logL_max=logL_max, p0=p0)
    
    # initialise chains
    pos = np.array(p) + np.array([0.1,0.1,0.1]) * np.random.randn(nwalkers, len(p))
    nwalkers, ndim = pos.shape

    inrange = np.where((logL >= logL_min) & (logL <= logL_max))[0]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, 
                                    args=(logL[inrange], LF_logL[inrange]))

    sampler.run_mcmc(pos, 1000, progress=True)
    
    return sampler


def plot_emcee(sampler, 
               labels = [r'$\alpha$', r'$\log_{10} L^\star$', r'$\log_{10} \phi^\star$'],
               discard=100, thin=30):
    
    # Chains
    samples = sampler.get_chain()

    fig, axes = plt.subplots(len(labels), figsize=(8, 4), sharex=True)
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.1)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    
    
    # Corner
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    fig = corner.corner(flat_samples, labels=labels, quantiles=[.16, .50, .84], show_titles=True)
    
    return

def plot_emcee_draws(sampler, logL, LF_logL, Ndraw=100,
                     labels = [r'$\alpha$', r'$\log_{10} L^\star$', r'$\log_{10} \phi^\star$'],
                     discard=100, thin=30,
                     xlim=(41,44), ylim=(1e-7, 1e-1)):
    
    """
    Plot draws from emcee
    """
    
    plt.annotate('best-fit: '+', '.join(f'{l}' for l in labels), xy=(0.05,0.25),xycoords='axes fraction')

    # input model LF
    plt.semilogy(logL, LF_logL, zorder=Ndraw, label='input')
    
    # scipy curve_fit
    p, cov = fit_schechter_scipy(logL, LF_logL)
    plt.plot(logL, schechter_logL(logL, *p), zorder=Ndraw+1, lw=2, ls='dashed', label='curve_fit')
    plt.annotate('curve_fit: '+', '.join(f'{param:.2f}' for param in p), xy=(0.05,0.1), xycoords='axes fraction')

    
    # emcee
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    # Plot N random draws from posterior
    for sample in flat_samples[np.random.randint(len(flat_samples), size=Ndraw)]:
        plt.plot(logL, schechter_logL(logL, *sample), 'k', alpha=0.1, lw=1)

    # Plot median parameters from posterior
    med_p = np.median(flat_samples, axis=0)
    plt.plot(logL, schechter_logL(logL, *med_p), zorder=Ndraw+2, lw=2, label='median emcee')
    plt.annotate('emcee: '+', '.join(f'{param:.2f}' for param in med_p), xy=(0.05,0.15), xycoords='axes fraction')

    plt.legend(loc='upper right')
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.xlabel(r'$\log_{10} L_\alpha/{\mathrm{erg \, s}^{-1}}$')
    plt.ylabel(r'$\phi[\mathrm{cMpc}^{-3}\, (\log_{10} L_\alpha)^{-1}]$')
    
    return

def get_emcee_medians(sampler, discard=100, thin=30):
    """
    Returns medians in array
    alpha, logLstar, logphistar
    """
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    medians = np.percentile(flat_samples, 50, axis=0)
    return medians

def get_emcee_params(sampler, discard=100, thin=30):
    """
    Prints

    alpha, alpha_l, alpha_u, logLstar, logLstar_l, logLstar_h, logphistar, logphistar_l, logphistar_h
    """
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    mcmc = np.percentile(flat_samples, [16, 50, 84], axis=0)
    q = np.diff(mcmc, axis=0)

    txt = '{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}'
    txt = txt.format(mcmc[1][0], q[0][0], q[1][0], 
                     mcmc[1][1], q[0][1], q[1][1], 
                     mcmc[1][2], q[0][2], q[1][2])

    return txt