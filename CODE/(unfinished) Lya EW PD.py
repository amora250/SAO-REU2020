#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.lines as mlines

## this is manipulated data from original code in README


def load_pW_pandas(lypW_filename): ##defined parameter as my_function then eventually paseed lnpW_file into it
    """
    Load table into pandas df
    """
    lnpW_tab_df = pd.read_csv(lypW_filename, skiprows=1, delim_whitespace=True)
    
    # Shuffle the column names to remove the '#' from the first column
    lnpW_tab_df.columns = np.roll(lnpW_tab_df.columns, -1)

    # Cut off the last (empty) column
    lnpW_tab_df = lnpW_tab_df.iloc[:, :-1]

    return lnpW_tab_df

# Pick xHI, Muv values to plot
xHI = 0.66 ##noticed that when xHI is changed, the file "...xHI=%.2f.txt" and thus the plot changes
xHI2= 0.05
xHI3= 0.36
xHI4= 0.87
xHI5= 0.01
#Muv = -21.0 ##commented this out from original code and adjusted plt.plot instead

# Load ln p(W) table
lnpW_file = '/Users/alexamia/Desktop/SAO REU 2020/M18_z=7.0_lnpWobs_Muv/ln_pWobs_xHI=%.2f.txt' % xHI
lnpW_tab = load_pW_pandas(lnpW_file) ## Passed lnpW_file to plot lnpW_tab using plt

lnpW_file2 = '/Users/alexamia/Desktop/SAO REU 2020/M18_z=7.0_lnpWobs_Muv/ln_pWobs_xHI=%.2f.txt' % xHI2
lnpW_tab2 = load_pW_pandas(lnpW_file2) 

lnpW_file3 = '/Users/alexamia/Desktop/SAO REU 2020/M18_z=7.0_lnpWobs_Muv/ln_pWobs_xHI=%.2f.txt' % xHI3
lnpW_tab3 = load_pW_pandas(lnpW_file3) 

lnpW_file4 = '/Users/alexamia/Desktop/SAO REU 2020/M18_z=7.0_lnpWobs_Muv/ln_pWobs_xHI=%.2f.txt' % xHI4
lnpW_tab4 = load_pW_pandas(lnpW_file4) 

lnpW_file5 = '/Users/alexamia/Desktop/SAO REU 2020/M18_z=7.0_lnpWobs_Muv/ln_pWobs_xHI=%.2f.txt' % xHI5
lnpW_tab5 = load_pW_pandas(lnpW_file5) 



# In[18]:


# Plot
plt.figure(figsize=(5,5))

##original README code was giving a syntax error with second part of plt.plot that had lnpW_tab[f'{Muv}'] 
## so I replaced it with the 2 Muv's used in Fig. 7 of Mason+18 and got the following graph below 

#plt.plot(lnpW_tab['W'],lnpW_tab['f{Muv:.1f}']) 

#Data with Muv= -18.0
plt.plot(lnpW_tab['W'],lnpW_tab['-18.0'],label='$\overline{x}_\mathrm{HI}=0.66$',color='blue') 
plt.plot(lnpW_tab2['W'], lnpW_tab2['-18.0'],label='$\overline{x}_\mathrm{HI}=0.05$', color='black')
plt.plot(lnpW_tab3['W'], lnpW_tab3['-18.0'],label='$\overline{x}_\mathrm{HI}=0.36$', color='green')
plt.plot(lnpW_tab4['W'], lnpW_tab4['-18.0'],label='$\overline{x}_\mathrm{HI}=0.87$', color='pink')
plt.plot(lnpW_tab5['W'], lnpW_tab5['-18.0'],label='$\overline{x}_\mathrm{HI}=0.01$', color='red')


L1 = mlines.Line2D([], [], label='$\overline{x}_\mathrm{HI}=0 (z=6)$', color='red')
L2 = mlines.Line2D([], [], label='$\overline{x}_\mathrm{HI}=0.05$', color='black')
L3 = mlines.Line2D([], [], label='$\overline{x}_\mathrm{HI}=0.36$', color='green')
L4 = mlines.Line2D([], [], label='$\overline{x}_\mathrm{HI}=0.66$',color='blue')
L5 = mlines.Line2D([], [], label='$\overline{x}_\mathrm{HI}=0.87$', color='pink')
FL = plt.legend(handles=[L1,L2,L3,L4,L5], loc='upper right',frameon=False)
# Add the legend manually to the current axes
ax = plt.gca().add_artist(FL)


#Data with Muv= -22.0
plt.plot(lnpW_tab['W'],lnpW_tab['-22.0'],linestyle='dashed',color='blue')
plt.plot(lnpW_tab2['W'], lnpW_tab2['-22.0'],linestyle='dashed', color='black')
plt.plot(lnpW_tab3['W'], lnpW_tab3['-22.0'],linestyle='dashed', color='green')
plt.plot(lnpW_tab4['W'], lnpW_tab4['-22.0'],linestyle='dashed', color='pink')
plt.plot(lnpW_tab5['W'], lnpW_tab5['-22.0'],linestyle='dashed', color='red')

Muv_line = mlines.Line2D([], [], color='black', label='$M_\mathrm{UV}=-18.0$')
Muv_line2 = mlines.Line2D([], [], color='black', label='$M_\mathrm{UV}=-22.0$', linestyle='dashed')
plt.legend(handles=[Muv_line,Muv_line2], loc='lower left',frameon=False)



plt.xlim(-10.,150.)
plt.ylim(-10.,0.)
plt.xlabel(r'$W$ [$\mathrm{\AA}$]')
plt.ylabel(r'$\ln{ p(W \;|\; \overline{x}_\mathrm{HI}, M_\mathrm{UV})}$')
plt.title(r'Ly-$\alpha$ EW Probability Distribution')







