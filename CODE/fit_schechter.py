import numpy as np
import os, sys
import matplotlib.pylab as plt
import pandas as pd
import scipy.interpolate
import emcee
import corner
import matplotlib as mpl
import multiprocessing
import itertools as it
import argparse            # argument managing

import run_LF as LF

plt.style.use(['default','seaborn-bright','seaborn-ticks'])
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

# ==============================================================================
# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
# ---- optional arguments ----
parser.add_argument("--logL_min", type=float, help="logL min for fitting, default = 41")
parser.add_argument("--logL_max", type=float, help="logL min for fitting, default = 43.8")
parser.add_argument("--num_proc", type=int, help="number of processors to use for multiprocessing, default = ncores - 1")
# ---- flags ------
parser.add_argument("--nosave", type=float, help="don't save plots, default = save")
# ==============================================================================


args = parser.parse_args()

num_proc = os.cpu_count() - 1
if args.num_proc:
    num_proc = args.num_proc
print(f'Running on {num_proc} cores')

logL_min = 41.
if args.logL_min:
    logL_min = args.logL_min

logL_max = 43.8
if args.logL_max:
    logL_max = args.logL_max

print(f'Fitting LF between {logL_min} - {logL_max}')

save = True
if args.nosave:
    save = False
    print('Not saving figures')

# ==============================================================================

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


def ln_likelihood(theta, logL, log_ndens):
    """
    log likelihood = -0.5 \Sum (data - model)^2
    
    use log data and model to make the likelihood a sensible number
    """
    lnlike = -0.5*np.nansum((log_ndens - ln_schechter_logL(logL, *theta))**2.)

    return lnlike
        

def ln_prior(theta):
    """
    uniform prior on log Schechter parameters
    """
    alpha, logLstar, logPhistar = theta
    if -4 < alpha < 0 and 40. < logLstar < 44. and -10 < logPhistar < -2:
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
                        p0=[-2., 43, -4.]):
    """
    Use scipy curve_fit to minimise chi2 = sum[ln(LF) - ln(LF_schechter)]
    
    Outputs:
        p = alpha, logLstar, logPhistar
    """
    
    log_LF_logL = np.log(LF_logL)
    
    inrange = np.where((logL >= logL_min) & (logL <= logL_max))[0]

    p, covar = scipy.optimize.curve_fit(ln_schechter_logL, logL[inrange], log_LF_logL[inrange], 
                                        p0=p0, bounds=([-6, 40, -10], [0, 45, -2]),
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
    log_LF_logL = np.log(LF_logL)    
    
    # run scipy to get starting position
    p, cov = fit_schechter_scipy(logL, LF_logL, logL_min=logL_min, logL_max=logL_max, p0=p0)
    
    # initialise chains
    pos = np.array(p) + np.array([0.5,0.5,0.5]) * np.random.randn(nwalkers, len(p))
    nwalkers, ndim = pos.shape

    inrange = np.where((logL >= logL_min) & (logL <= logL_max))[0]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, 
                                    args=(logL[inrange], log_LF_logL[inrange]))

    sampler.run_mcmc(pos, nsteps, progress=True)
    
    return sampler


# --------------------------------------------------------
# Plotting

def plot_emcee(sampler, 
               labels = [r'$\alpha$', r'$\log_{10} L^\star$', r'$\log_{10} \phi^\star$'],
               discard=100, thin=1, truths=None,
               save=False, plotname='emcee.png'):
    
    # Chains
    samples = sampler.get_chain()

    fig, axes = plt.subplots(len(labels), figsize=(8, 4), sharex=True)
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.1)
        ax.axvline(discard)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    if save:
        plt.savefig(plotname.replace('.png','_chains.png'), bbox_inches='tight')  
    
    # Corner
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    fig = corner.corner(flat_samples, labels=labels, quantiles=[.16, .50, .84], truths=truths, show_titles=True)
    
    if save:
        plt.savefig(plotname.replace('.png','_corner.png'), bbox_inches='tight')    
    
    return

def plot_emcee_draws(sampler, logL, LF_logL, Ndraw=100,
                     labels = [r'$\alpha$', r'$\log_{10} L^\star$', r'$\log_{10} \phi^\star$'],
                     discard=100, thin=1,
                     xlim=(41,44), ylim=(1e-7, 1e-1),
                     save=False, plotname='emcee.png'):
    
    """
    Plot draws from emcee
    """
    
    logL, LF_logL = logL[np.isfinite(logL)], LF_logL[np.isfinite(logL)]
    
    plt.annotate('best-fit: '+', '.join(f'{l}' for l in labels), xy=(0.05,0.25),xycoords='axes fraction')

    # input model LF
    plt.semilogy(logL, LF_logL, ls='dashed', zorder=Ndraw+10, label='input')
    
    # scipy curve_fit
    p, cov = fit_schechter_scipy(logL, LF_logL)
    plt.plot(logL, np.exp(ln_schechter_logL(logL, *p)), zorder=Ndraw+1, lw=2, ls='dotted', label='curve_fit')
    plt.annotate('curve_fit: '+', '.join(f'{param:.2f}' for param in p), xy=(0.05,0.1), xycoords='axes fraction')

    
    # emcee
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    # Plot N random draws from posterior
    for sample in flat_samples[np.random.randint(len(flat_samples), size=Ndraw)]:
        plt.plot(logL, np.exp(ln_schechter_logL(logL, *sample)), 'k', alpha=0.1, lw=1)

    # Plot median parameters from posterior
    med_p = np.median(flat_samples, axis=0)
    plt.plot(logL, np.exp(ln_schechter_logL(logL, *med_p)), zorder=Ndraw+2, lw=2, label='emcee median')
    plt.annotate('emcee: '+', '.join(f'{param:.2f}' for param in med_p), xy=(0.05,0.15), xycoords='axes fraction')

    plt.legend(loc='upper right')
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.xlabel(r'$\log_{10} L_\alpha/{\mathrm{erg \, s}^{-1}}$')
    plt.ylabel(r'$\phi[\mathrm{cMpc}^{-3}\, (\log_{10} L_\alpha)^{-1}]$')

    if save:
        plt.savefig(plotname.replace('.png','_LFdraws.png'), bbox_inches='tight')
                    
    return

# --------------------------------------------------------
# Get params

def get_emcee_medians(sampler, discard=100, thin=1):
    """
    Returns medians in array
    alpha, logLstar, logphistar
    """
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    medians = np.percentile(flat_samples, 50, axis=0)
    return medians

def get_emcee_params(sampler, discard=100, thin=1):
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


# --------------------------------------------------------
# Run fits wrapper

def schechter_fit(LFdict, zval_test, xHI_test, logL_min=41., logL_max=44., 
                  plot=True, plot_chains=True, save=True):
    
    #Call LF function
    log10_LF = LFdict[(zval_test, xHI_test)]

    # Run emcee to get posterior samples
    sampler = fit_schechter_emcee(LF.log10_lg[LF.log10_lg > 0.], log10_LF[LF.log10_lg > 0.], 
                                  logL_min=logL_min, logL_max=logL_max)
    
    plotname = f'../data/schechter_runs/logL={logL_min}-{logL_max}_z={zval_test}_xHI={xHI_test}.png'
    
    if plot:
        if plot_chains:
            plt.figure()
            plot_emcee(sampler, plotname=plotname, save=save)
            plt.close()              
        
        plt.figure()
        plot_emcee_draws(sampler, LF.log10_lg, log10_LF, 
                            xlim=(logL_min-0.5, logL_max+0.5),
                            plotname=plotname, save=save)
        plt.close('all')

    medians = get_emcee_medians(sampler)
    
    del sampler
    
    return medians

# ==============================================================================

if __name__ == '__main__':
    
    multiprocessing.set_start_method('spawn')

    LFdict = np.load('../data/allLFvals.npy', allow_pickle=True).item()
    z   = np.unique(np.array([k[0] for k in list(LFdict.keys())]))
    xHI = np.unique(np.array([k[1] for k in list(LFdict.keys())]))
    print('Loaded LF dictionary')
    print(z)

    def input_params(LFdict, z, xHI, logL_min, logL_max, plot=True, plot_chains=True, save=True):
        for pair in it.product(z, xHI):
            yield (LFdict, *pair, logL_min, logL_max, plot, plot_chains, save)

    input_params_iter = input_params(LFdict, z, xHI, logL_min, logL_max, save=save)

    with multiprocessing.Pool(processes=num_proc) as pool:
        results = pool.starmap(schechter_fit, input_params_iter)

    res_dict = {}
    for i, x in enumerate(input_params_iter):
        res_dict[(x[1], x[2])] = results[i]

    np.save(f'../data/Schechter_params_logL={logL_min}-{logL_max}.npy', res_dict)