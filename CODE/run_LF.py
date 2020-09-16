import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import os,sys
import fit_schechter as fs
import glob
import scipy
from astropy.cosmology import Planck15 as P15
from astropy import units as u
from astropy import constants as const
from scipy import interpolate
from scipy.optimize import minimize, differential_evolution, basinhopping


plt.style.use(['default','seaborn-colorblind','seaborn-ticks'])
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

#Constants
Muv_array = np.array([-22.0,-16.0]) #chosen Muv to test lum_lya vs Muv similar to EW vs Muv
beta = -2.0 #usually -2 for high z galaxies as per spectrum as power law
pc_10 = 10 * u.pc #1 pc to Mpc
wl_lya = 1216 * u.Angstrom #angstrom
wl_uv = 1500 * u.Angstrom #angstrom
f0 = 3.631e-20 * (u.erg/u.s) * (u.cm**(-2)) * (u.Hz**(-1)) #flux_0 in erg s^-1 cm^-2 Hz^-1
c = const.c #speed of light
lum_grid = np.logspace(38,44, num = 200) #shape = (50,)
log10_lg = np.log10(lum_grid) #log10 luminosity grid in order to plot it on log10 scale similar to past works
Muv_grid = np.round(np.arange(-24, -12, 0.1),1)

xHI_list = np.array([0.01, 0.02, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22, 0.25, 0.29, 
                0.32, 0.36, 0.39, 0.42, 0.45, 0.49, 0.52, 0.55, 0.58, 0.61, 0.64,
                0.66, 0.69, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.87,
                0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95])



#Functions
def insensitive_glob(pattern):
    """
    Case insensitive find file names
    """
    def either(c):
        return '[%s%s]'%(c.lower(),c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either,pattern)))

#Function used to load files
def load_uvf_pandas(ufl_filename): 
    """
    Load table into pandas df
    """
    uvf_tab_df = pd.read_csv(ufl_filename, skiprows=1, delim_whitespace=True)
    
    # Shuffle the column names to remove the '#' from the first column
    uvf_tab_df.columns = np.roll(uvf_tab_df.columns, -1)

    # Cut off the last (empty) column
    uvf_tab_df = uvf_tab_df.iloc[:, :-1]

    return uvf_tab_df


#Call UV LF data files
LFz_data_dir = os.environ['LYA_DATA_DIR']+'data/models/' #calls directory
LFz_dir = LFz_data_dir+'MTT15_UVLF/LF_pred/' #inside models folder call MTT15_UVLF/LF_pred/ folder
LFz_files = sorted(insensitive_glob(LFz_dir+'LF_pred_z*.txt')) 
#calls each file in modelled data * will be replaced with corresponding zval

#Calling EW files and their Muv values
pW_data_dir = os.environ['LYA_DATA_DIR']+'data/models/' 
pW_dir = pW_data_dir+'M18_z=7.0_lnpWobs_Muv/' #inside models folder call M18_z=7.0_lnpWobs_Muv folder
pW_files = sorted(insensitive_glob(pW_dir+'ln_pWobs_*.txt')) #calls each file in modelled data * will be replaced with corresponding xHI

#Calling Konno data file
K_data_dir = os.environ['LYA_DATA_DIR']+'data/models/' 
K_dir = pW_data_dir+'Lya_LF_Konno/' 
K_files = sorted(insensitive_glob(K_dir+'Lya_LF_Konno_z*.txt')) 

#Calling Ouchi data file
O_data_dir = os.environ['LYA_DATA_DIR']+'data/models/' 
O_dir = pW_data_dir+'Lya_LF_Ouchi/' 
O_files = sorted(insensitive_glob(O_dir+'Lya_LF_Ouchi_z*.txt')) 

#Calling Santos data file
S_data_dir = os.environ['LYA_DATA_DIR']+'data/models/' 
S_dir = pW_data_dir+'Lya_LF_Santos/' 
S_files = sorted(insensitive_glob(S_dir+'Lya_LF_Santos_z*.txt')) 

#Calling Luminosity Density info
LD_data_dir = os.environ['LYA_DATA_DIR']+'data/models/' 
LD_dir = pW_data_dir+'LD/' 
LD_files = sorted(insensitive_glob(LD_dir+'LD_info.txt')) 

#Calling Shibuya data file
Sh_data_dir = os.environ['LYA_DATA_DIR']+'data/models/' 
Sh_dir = pW_data_dir+'Lya_LF_Shibuya/' 
Sh_files = sorted(insensitive_glob(S_dir+'Lya_LF_Shibuya_z*.txt')) 





#Plot UV LF values vs interpolated Muv that is the same as EW 
def plot_UV_LF(Muv_EW,new_ndens):
    '''
    Plots UV LF for a interpolated EW Muv grid
    
    '''
    plt.semilogy(LF_tab['Muv'],LF_tab['ndens'],label = 'z = %.1f'%zval_test) #UV LF values (181 Muv vals)
    plt.plot(Muv_EW,new_ndens,'o')#EW Values along UV LF plot (61 Muv vals)
    #Muv values for EW are restricted in range, we'll have to extend range, use same EW PD > -23 and < -17 
    plt.legend()
    plt.xlabel('$M_\mathrm{UV}$')
    plt.ylabel(r'$\phi(M_\mathrm{UV})\ Mpc^-3$')
    plt.show()
    return

def plot_jvsMuv(jacobian, Muv_EW, zval_test):
    '''
    Plots relationship between lya/EW jacobian vs Muv values
    
    '''
#Shows a positive linear relationship b/t Muv and jacobian, (fd_uv vs Muv shows neg. LR)
#As Muv becomes fainter, value for jacobian increases, they're inversely proportional?

    plt.semilogx(jacobian, Muv_EW, label = 'z = %.1f'%zval_test)     
    plt.legend()
    plt.xlabel(r'$\partial EW/\partial L_\alpha$, [$\mathrm{\AA} cm^2 s/erg Mpc^2$]')
    plt.ylabel('$M_\mathrm{UV}$')
    plt.title('Jacobian vs $M_\mathrm{UV}$ for a Given Redshift ')
    return

def konno_data_plt(zval_test, plot = False, mean = False):
    '''
    Plots observational Lya LF data from Konno+18,+14 to fit model
    
    '''
    K_file = sorted(insensitive_glob(K_dir+f'Lya_LF_Konno_z*{zval_test}.txt'))[0]
    Ko_tab = load_uvf_pandas(K_file)
    
   
    yerr_l = 10**(Ko_tab['log(ndens)']) - 10**(Ko_tab['ndens_l2']) 
    yerr_u = 10**(Ko_tab['ndens_u2']) - 10**(Ko_tab['log(ndens)'])
    yerror = np.array([yerr_l,yerr_u])
    
    Ko_ndens = np.array(10**(Ko_tab['log(ndens)']))
    Ko_L = np.array(Ko_tab['log(L)'])
    yerror_mean = np.mean(yerror,axis=0)
    

    if plot==True:
        if zval_test == 5.7:
            my_color = 'blue'
            label2 = 'Konno+18'
        elif zval_test == 6.6:
            my_color = 'red'
            label2 = 'Konno+18'
        else:
            my_color = 'orange'
            label2 = 'Konno+14'


        plt.semilogy(Ko_tab['log(L)'], 10**(Ko_tab['log(ndens)']),color=my_color, alpha=0.5, marker='o', lw=0)
        plt.errorbar(Ko_tab['log(L)'], 10**(Ko_tab['log(ndens)']),yerr=yerror, fmt = ' ',capsize=5, color=my_color)
        
    if mean == True:
        return Ko_L, Ko_ndens, yerror_mean
    else:
        return Ko_L, Ko_ndens, yerror

def santos_data_plt(zval_test, plot = False):
    '''
    Plots observational Lya LF data from Santos+16 to fit model
    
    '''
    S_file = sorted(insensitive_glob(S_dir+f'Lya_LF_Santos_z*{zval_test}.txt'))[0]
    Sa_tab = load_uvf_pandas(S_file)
    
   
    yerr_l = 10**(Sa_tab['log(ndens)']) - 10**(Sa_tab['ndens_l2']) 
    yerr_u = 10**(Sa_tab['ndens_u2']) - 10**(Sa_tab['log(ndens)'])
    yerror = np.array([yerr_l,yerr_u])
    
    Sa_ndens = np.array(10**(Sa_tab['log(ndens)']))
    Sa_L = np.array(Sa_tab['log(L)'])
    yerror_mean = np.mean(yerror,axis=0)
    

    if plot==True:
        if zval_test == 5.7:
            my_color = 'blue'
            label2 = 'Santos+16'
        elif zval_test == 6.6:
            my_color = 'red'
            label2 = 'Santos+16'


        plt.semilogy(Sa_tab['log(L)'], 10**(Sa_tab['log(ndens)']),color=my_color, alpha=0.5, marker='*', lw=0)
        plt.errorbar(Sa_tab['log(L)'], 10**(Sa_tab['log(ndens)']),yerr=yerror, fmt = ' ',capsize=5, color=my_color)
    return Sa_L, Sa_ndens, yerror_mean

def ouchi_data_plt(zval_test, plot = False, mean = False):
    '''
    Plots observational Lya LF data from Ouchi+08,+10 to fit model
    
    '''
    O_file = sorted(insensitive_glob(O_dir+f'Lya_LF_Ouchi_z*{zval_test}_ndens.txt'))[0]
    Ou_tab = load_uvf_pandas(O_file)

    yerror = np.array([Ou_tab['error_l'],Ou_tab['error_u']])
    
    Ou_ndens = np.array((Ou_tab['ndens']))
    Ou_L = np.array(Ou_tab['log(L)'])
    yerror_mean = np.mean(yerror,axis=0)


    if plot==True:
        if zval_test == 5.7:
            my_color = 'blue'
            label2 = 'Ouchi+08'
        elif zval_test == 6.6:
            my_color = 'red'
            label2 = 'Ouchi+10'



        plt.semilogy(Ou_tab['log(L)'], (Ou_tab['ndens']),color=my_color, alpha=0.5, marker='^', lw=0)
        plt.errorbar(Ou_tab['log(L)'], (Ou_tab['ndens']),yerr=[Ou_tab['error_l'],Ou_tab['error_u']], fmt = ' ',capsize=5, color=my_color)
    
    if mean==True:
        return Ou_L, Ou_ndens, yerror_mean
    else:
        return Ou_L, Ou_ndens, yerror
    
def shibuya_data_plt(zval_test, plot = False, mean = False):
    '''
    Plots observational Lya LF data from Shibuya+12 to fit model
    
    '''
    Sh_file = sorted(insensitive_glob(Sh_dir+f'Lya_LF_Shibuya_z*{zval_test}.txt'))[0]
    Sh_tab = load_uvf_pandas(Sh_file)

    
    yerr_l = (Sh_tab['ndens']) - (Sh_tab['error_l']) 
    yerr_u = (Sh_tab['error_u']) - (Sh_tab['ndens'])
    yerror = np.array([yerr_l,yerr_u])
    
    xerr_l = (Sh_tab['log(L)']) - (Sh_tab['log(L)_l']) 
    xerr_u = (Sh_tab['log(L)_u']) - (Sh_tab['log(L)'])
    xerror = np.array([xerr_l,xerr_u])
    
    Sh_ndens = np.array((Sh_tab['ndens']))
    Sh_L = np.array(Sh_tab['log(L)'])
    
    yerror_mean = np.mean(yerror,axis=0)


    if plot==True:
        if zval_test == 7.3:
            my_color = 'orange'
            label2 = 'Shibuya+12'

        plt.semilogy(Sh_tab['log(L)'], (Sh_tab['ndens']),color=my_color, alpha=0.5, marker='p', lw=0)
        plt.errorbar(Sh_tab['log(L)'], (Sh_tab['ndens']),yerr=yerror, xerr = xerror, fmt = ' ',capsize=5, color=my_color)
        #xerr = [Sh_tab['log(L)_l'],Sh_tab['log(L)_u']],
    
    if mean==True:
        return Sh_L, Sh_ndens, yerror_mean
    else:
        return Sh_L, Sh_ndens, yerror


def log10_LF_plot(log10_LF,zval_test,xHI_test,plot = False):
    '''
    This is used to plot the model of the Lya LF at different xHI and z values
    '''
    plt.semilogy(log10_lg, log10_LF,label = '$\overline{x}_\mathrm{HI}$ = %.2f'%xHI_test)
#     plt.semilogy(log10_lg, log10_LF,label = 'z = %.2f'%zval_test)

    return 

def LvsPLya(Muv_array, zval_test, lum_lya, norm_pLya, new_pLya):
    '''
    Still needs to be fixed, new plot created with old code formatting!
    
    Plots relationship between Lum. and P(Lum.) for different Muv and xHI values
    '''
    
    for mm,Muv in enumerate (Muv_array):
            if mm == 0: #first item in Muv_array, i.e. -18.0
                ls = 'solid'
                label = 'z = %.1f'%zval_test
            elif mm == 1: #second item in Muv_array, i.e. -22.0 (will not show on xHI legend)
                ls = 'dashed'
                label = None
            elif mm == 2: # Third item in Muv_array, i.e. -16.0
                ls = 'dashdot'
                plt.loglog(lum_lya[:,mm], norm_pLya[:,mm], ls=ls, color = my_color, label = label) #[:,mm] gets corresponding column than row [mm]
                plt.plot(lum_grid, new_pLya[:,mm], ls=ls, color = 'blue')

    leg_zval = plt.legend(frameon=False, handletextpad=0.5)
    plt.gca().add_artist(leg_zval)
    #This is the legend for Muv values, -18.0 is solid, -22.0 is dashed, -16.0 is dash-dot
    line2 = mlines.Line2D([], [], color='k', label=r'$M_\mathrm{UV} = %.1f$' % Muv_array[0])
    line3 = mlines.Line2D([], [], color='k', ls='dashed', label=r'$M_\mathrm{UV} = %.1f$' % Muv_array[1])
    line1 = mlines.Line2D([], [], color='k', ls='dashdot', label=r'$M_\mathrm{UV} = %.1f$' % Muv_array[2])
    plt.legend(handles=[line1, line2, line3], loc='lower left', frameon=False, handletextpad=0.5, handlelength=1.5)

    plt.ylim(1e-50,1e-44)
    plt.xlabel(r'${\mathrm{L_\alpha}}$, [$erg/s$]')
    plt.ylabel(r'${ P (L_\alpha \;|\; M_\mathrm{UV})}$')
    return


#Defines function for lya Luminosity probability
def make_pL_Lya(zval_test, xHI_test):
    
    #Calling UV, EW, Konno files to obtain z and xHI values
    pW_file = sorted(insensitive_glob(pW_dir+f'ln_pWobs_*{xHI_test:.2f}.txt'))[0]
    
    
    #Load in xHI value file
    pW_tab = load_uvf_pandas(pW_file)
    
    #Get Muv values from file as an array to use
    Muv_EW = np.array([float(Muv_val) for Muv_val in pW_tab.columns[1:]])
    
    #Defines luminosity distance
    d_l = P15.luminosity_distance(zval_test) 
    
    #Define apparent magnitude equation
    def muv(Muv,d_l,zval_test): 
        p1 = Muv
        p2 = 5*(np.log10((d_l/pc_10).to(u.pc/u.pc))) #convert d_l/pc_10 to pc units and then dimensionless
        ans = p1 + p2
        return ans    
    muv = muv(Muv_grid,d_l,zval_test)
    
    #Flux density of UV continuum at Lya wavelength from Muv for given zvals
    fd_uv = f0 * (10**(-0.4*muv)) *(c/(wl_lya**2)) *((wl_lya/wl_uv)**(beta+2.0))
    fd_units = fd_uv.to(u.erg/u.s* (u.cm**(-2))/u.Angstrom) ##use this to show units
    
    #Jacobian - partial EW / partial Lya Luminosity for given zvals 
    jacobian = 1/((4*np.pi*d_l**2.)*fd_units)
    
    # Flux_Lya
    f_lya = np.outer(pW_tab['W'],fd_units) * u.Angstrom
    
    #Lya luminosity 
    lum_lya = (f_lya * (4*np.pi*d_l**2.)).to(u.erg/u.s)
    
    #Drops first column of EW values
    new_pW_tab = np.exp(pW_tab.drop('W',axis=1))
    
    #EW values for each Muv in EW file
    pEW_vals = np.array(new_pW_tab) 
    
    # Create p(EW | Muv) on bigger Muv grid, assuming the EW distribution 
    # doesn't evolve with magnitude for very bright or very faint galaxies
    pEW_vals_Muv_grid = np.zeros((pEW_vals.shape[0],len(Muv_grid)))
    
    # Index of brightest Muv_EW galaxy in Muv_grid
    m_index = np.where(Muv_grid == np.min(Muv_EW))[0][0]
    for mm, Muv in enumerate(Muv_grid):
        # For faint galaxies, use p(EW | Muv =-17)
        if Muv > np.max(Muv_EW):
            pEW_vals_Muv_grid[:,mm] = pEW_vals[:,-1]
        # For bright galaxies, use p(EW | Muv =-22)
        elif Muv < np.min(Muv_EW):
            pEW_vals_Muv_grid[:,mm] = pEW_vals[:,0]
        # Otherwise use the model p(EW | Muv)
        else:
            pEW_vals_Muv_grid[:,mm] = pEW_vals[:,mm-m_index]
    
    
    # P(Lya|Muv)
    pLya = jacobian * pEW_vals_Muv_grid
    
    #Normalizes pLya to correctly plot lum_lya vs pLya
    A1 = pLya[0] #first term of p(L) integral, where these lum = 0 (the height in y)(-inf to 0)
    integral = np.trapz(pLya[1:].T,lum_lya.value[1:].T) #second term for lum>0 (L>0 to inf) transposed to correct matrix 
    sum_int = A1 + integral #This is whole integral of p(L)dL
    norm_pLya = pLya / sum_int #new normalized pLya, divide original pLya values by integral to normalize
    new_A1 = norm_pLya[0]
    integral2 = np.trapz(norm_pLya[1:].T,lum_lya.value[1:].T)
    new_sum = new_A1 + integral2 #verifies that new normalized integral adds up to 1 
    assert(new_sum.value.all() == 1.)
    
    
    #Define an empty matrix in order to fill later with luminosity grid values and Muv values
    new_pLya = np.zeros((len(lum_grid), len(Muv_grid))) 
    
    for mm,Muv in enumerate (Muv_grid):
        #Interpolating pLya and ndens values into a 1d array
        LF_interp1 = interpolate.interp1d(lum_lya[:,mm],norm_pLya[:,mm],fill_value=0., bounds_error=False)
        new_pLya[:,mm] = LF_interp1(lum_grid) #column values of pLya
    
    return Muv_grid, new_pLya


#Defining lya LF function and all necessary eqs needed 

def make_lya_LF(zval_test, xHI_test, F=1., plot=False, log=True):
    #Calling UV, EW, Konno files to obtain z and xHI values
    LFz_file = sorted(insensitive_glob(LFz_dir+ f'LF_pred_z{zval_test}.txt'))[0] 
    
    # Make p(L_Lya | Muv)
    Muv_grid, new_pLya = make_pL_Lya(zval_test, xHI_test)
    
    #Load in z value file
    LF_tab = load_uvf_pandas(LFz_file) 
    
    
    #Interpolating UV LF Muv and ndens values into a 1d array
    LF_interp = interpolate.interp1d(LF_tab['Muv'],LF_tab['ndens']) #old values
    new_ndens = LF_interp(Muv_grid)

        
    #Product of UVLF ndens values * pLya in new luminosity grid
    product_LF = new_ndens * new_pLya 


    #Integral of this product = Lya LF (missing fudge factor corrections)
    lya_LF =np.trapz(product_LF, x = Muv_grid) 


    #Log10 of Lya LF is lya_LF value * Jacobian - partial L / partial log10(L)
    log10_LF = lya_LF*np.log(10)*lum_grid
    
    
        
    
    ## CALLING FUNCTIONS 
    
    #Plotting Information
    if plot == True:
        if zval_test == 5.7:
            
            #Plot Konno info 
            konno_data_plt(zval_test, plot = True, mean = False)

            #Plot Ouchi info
            ouchi_data_plt(zval_test, plot = True, mean = False)
        elif zval_test == 6.6:
            
            #Plot Konno info 
            konno_data_plt(zval_test, plot = True, mean = False)

            #Plot Ouchi info
            ouchi_data_plt(zval_test, plot = True, mean = False)            
        
        elif zval_test == 7.3:
            #Plot Konno info 
            konno_data_plt(zval_test, plot = True, mean = False)
            
            #Plot Shibuya info
            shibuya_data_plt(zval_test, plot = True, mean = False)

#         #Jacobian vs Muv plot info
#         plot_jvsMuv(jacobian, Muv_EW, zval_test)

#         #Plot Lum vs PLum info
#         LvsPLya(Muv_array, zval_test, lum_lya, norm_pLya, new_pLya)

        #LF vs Konno plot info
        log10_LF_plot(log10_LF,zval_test,xHI_test, plot = True)
        

    
    if log == True:
        return F*log10_LF
    else:
        return F*lya_LF


    
#Minimization of xHI
#Line models used for z = 6.6, 7.3

def xHI_model(xHI, obs_L, zval=6.6):
    """
    Evaluate a straight line model at the input Konno luminosity values.
    
    Parameters
    ----------
    F : list, array
        This should be a length-2 array or list containing the 
        parameter values (a, b) for the (slope, intercept).
    new_phi_Li : numeric, list, array
        The coordinate values.
        
    Returns
    -------
    new_phi_Li : array
        The computed y values at each input x.
    """
    #Interpolating Konno and lum grid 
    xHI_LF_calibrate = make_lya_LF(zval_test = zval, xHI_test = xHI, F = 0.615, plot=False,log=True)   
    LF_interp = interpolate.interp1d(log10_lg, xHI_LF_calibrate)
    new_phi_Li = LF_interp(obs_L)
    
    return new_phi_Li

def xHI_weighted_squared_deviation(xHI,z = 6.6):
    """
    Chi = Konno, Ouchi ndens values - our ndens values at corresponding Konno, Ouchi lum grid * xHI model WRT xHI, obs. values, z
    Compute the weighted squared deviation between the data 
    (x, y, y_err) and the model points computed with the input 
    parameters (F).
    """
    if type(xHI) == float:
    #print singular value
        Ko_L, Ko_ndens, yerror = konno_data_plt(zval_test= z, plot = False, mean = False) #Defined here rather than as func. parameters
        xHI_chi_Ko = (Ko_ndens - xHI_model(xHI, obs_L = Ko_L, zval=z)) / yerror

        if z == 6.6:
            
            Ou_L, Ou_ndens, yerror = ouchi_data_plt(zval_test=z, plot = False, mean = False) #Defined here rather than as func. parameters
            xHI_chi_Ou = (Ou_ndens - xHI_model(xHI, obs_L = Ou_L,zval=z)) / yerror


            return np.sum(xHI_chi_Ko**2) + np.sum(xHI_chi_Ou**2)
        else:
            return np.sum(xHI_chi_Ko**2) 
    else:
        #prints list of values
        chi2=[]
        for x in xHI:
            #Finding nearest neighbor to values in xHI array 

            idx = (np.abs(xHI_list - x)).argmin()
            xHI_grid_match = xHI_list[idx]

            
            Ko_L, Ko_ndens, yerror= konno_data_plt(zval_test= z, plot = False, mean = False) #Defined here rather than as func. parameters
            xHI_chi_Ko = (Ko_ndens - xHI_model(xHI_grid_match, obs_L = Ko_L, zval=z)) / yerror

            if z == 6.6:
                Ou_L, Ou_ndens, yerror = ouchi_data_plt(zval_test=z, plot = False, mean = False) #Defined here rather than as func. parameters
                xHI_chi_Ou = (Ou_ndens - xHI_model(xHI_grid_match, obs_L = Ou_L,zval=z)) / yerror


                chi2.append(np.sum(xHI_chi_Ko**2) + np.sum(xHI_chi_Ou**2))
            else:
                chi2.append(np.sum(xHI_chi_Ko**2))
        return np.array(chi2)
    
    