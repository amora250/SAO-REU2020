#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.lines as mlines

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

#Define variables
zval = 0.0
zval1 = 0.3
zval2 = 2.0
zval3 = 3.8
zval4 = 4.9
zval5 = 5.9
zval6 = 6.8
zval7 = 7.9
zval8 = 9.0
zval9 = 10.4
zval10 = 11.1
zval11 = 12.0
zval12 = 14.0
zval13 = 16.0

# Load UV LF tables
uvf_file = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval
uvf_tab = load_uvf_pandas(uvf_file) 

uvf_file1 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval1
uvf_tab1 = load_uvf_pandas(uvf_file1) 

uvf_file2 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval2
uvf_tab2 = load_uvf_pandas(uvf_file2) 

uvf_file3 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval3
uvf_tab3 = load_uvf_pandas(uvf_file3) 

uvf_file4 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval4
uvf_tab4 = load_uvf_pandas(uvf_file4) 

uvf_file5 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval5
uvf_tab5 = load_uvf_pandas(uvf_file5) 

uvf_file6 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval6
uvf_tab6 = load_uvf_pandas(uvf_file6) 

uvf_file7 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval7
uvf_tab7 = load_uvf_pandas(uvf_file7) 

uvf_file8 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval8
uvf_tab8 = load_uvf_pandas(uvf_file8) 

uvf_file9 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval9
uvf_tab9 = load_uvf_pandas(uvf_file9) 

uvf_file10 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval10
uvf_tab10 = load_uvf_pandas(uvf_file10) 

uvf_file11 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval11
uvf_tab11 = load_uvf_pandas(uvf_file11) 

uvf_file12 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval12
uvf_tab12 = load_uvf_pandas(uvf_file12) 

uvf_file13 = '/Users/alexamia/Desktop/SAO REU 2020/UVLFM_MASON/LF_pred/LF_pred_z%.1f.txt' % zval13
uvf_tab13 = load_uvf_pandas(uvf_file13) 
inv = np.log(uvf_tab['ndens'])
inv1 = np.log(uvf_tab1['ndens'])


# In[72]:


#Plotting
fig = plt.figure(figsize=(5,10))
axs= fig.subplots(ncols=1,nrows=2, gridspec_kw={'hspace': 0},sharex=True, sharey=True)
plt.plot(uvf_tab['Muv'],inv,label='z~0',color='blue') 
plt.plot(uvf_tab1['Muv'],inv1,label='z~0.3',color='green') 

plt.xlim(-24.,-14.)
#plt.ylim(1e-8,1e-2)


# In[ ]:





# In[ ]:




