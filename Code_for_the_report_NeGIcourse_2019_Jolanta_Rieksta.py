#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dask.distributed import Client

client = Client("tcp://127.0.0.1:42930")
client


# # Import packages

# In[2]:


from pydap.client import open_dods, open_url
from netCDF4 import num2date
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
get_ipython().run_line_magic('matplotlib', 'inline')
import cartopy.crs as ccrs
import cartopy as cy
import matplotlib.path as mpath
from pydap.client import open_dods, open_url
from netCDF4 import num2date
import pandas as pd
import sys
import glob
import pyaerocom as pya
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')
pya.change_verbosity('critical', log=pya.const.print_log) # don't output warnings
pya.__version__


# # Load data

# In[3]:


# Isoprene Volume Mixing Ratio  (mol mol-1) NorESM model (historical run, years [1980 - 2014], lev=0, latlims [60:90])
path='/home/bd15084e-2d25b7-2d47db-2dade3-2daea695ce03d8/noresm_xr1.nc' 
noresm_isop = xr.open_dataset(path)


# In[4]:


# Isoprene Volume Mixing Ratio  (mol mol-1)  UKESM model (historical run, years [1980 - 2014], lev=0, latlims [60:90])
path='/home/bd15084e-2d25b7-2d47db-2dade3-2daea695ce03d8/ukesm_xr1.nc' 
ukesm_isop = xr.open_dataset(path)


# In[5]:


# Total Emission Rate of Isoprene (kg m-2 s-1) NorESM model (historical run, years [1980 - 2014], lev=0, latlims [60:90])
path='/home/bd15084e-2d25b7-2d47db-2dade3-2daea695ce03d8/noresm_emis1.nc' 
noresm_emis = xr.open_dataset(path)


# In[6]:


# Total Emission Rate of Isoprene (kg m-2 s-1) UKESM model (historical run, years [1980 - 2014], lev=0, latlims [60:90])
path='/home/bd15084e-2d25b7-2d47db-2dade3-2daea695ce03d8/ukesm_emiisop_4.nc' 
ukesm_emis = xr.open_dataset(path)


# # Plot NorthPolarStereo of Isoprene Volume Mixing Ratio (mol mol-1) for seasons for each of the models and obtain the diffrences between the models. 

# In[7]:


def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection':projection}, **kwargs)

def add_map_features(ax):
    ax.coastlines()
    gl = ax.gridlines()
    ax.add_feature(cy.feature.BORDERS);
    gl = ax.gridlines()#draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False


# In[8]:



def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)


# ### NorESM model Isoprene Volume Mixing Ratio  (mol mol-1) 

# In[9]:


#Group data by the season and obtain the mean from (1980-2014)
ds_seas = noresm_isop.groupby('time.season').mean()
# Make the axis with NorthPolarStereo() projection:
fig, axs = sp_map(2,2, projection=ccrs.NorthPolarStereo(), figsize=[10,10] )
lat_lims = [60,90]
season_l = ds_seas['season'].values
print(season_l)
for ax, seas in zip(axs.flatten(),season_l):
    _ds = ds_seas.sel(season=seas)['isop'].where(ds_seas['lat']>lat_lims[0])#
    _ds.attrs['units']='mol/mol'; _ds.attrs['long_name'] = 'Isoprene'
    _ds.plot(ax=ax, cmap=plt.get_cmap('Reds'),transform=ccrs.PlateCarree())
    ax.set_title('NorESM: '+ seas )
    polarCentral_set_latlim(lat_lims, ax)
    add_map_features(ax)
   
plt.show()


# **Figure 1.** Seasonal average (1980-2014) isoprene concentrations NorESM. **DJF**= December, January, February; **MAM** = March, April, May; **JJA** = June, July, August and  **SON**= September, Oktober, November. 

# ### UKESM model Isoprene Volume Mixing Ratio  (mol mol-1) 
# 

# In[10]:


#Group data by the season and obtain the mean from (1980-2014)
ds_seas = ukesm_isop.groupby('time.season').mean()
fig, axs = sp_map(2,2, projection=ccrs.NorthPolarStereo(), figsize=[10,10] )
lat_lims = [60,90]
season_l = ds_seas['season'].values
print(season_l)
for ax, seas in zip(axs.flatten(),season_l):
    _ds = ds_seas.sel(season=seas)['isop'].where(ds_seas['lat']>lat_lims[0])#
    _ds.attrs['units']='mol/mol'; _ds.attrs['long_name'] = 'Isoprene'
    _ds.plot(ax=ax, cmap=plt.get_cmap('Reds'),transform=ccrs.PlateCarree())
    ax.set_title('UKESM: '+ seas)
    polarCentral_set_latlim(lat_lims, ax)
    add_map_features(ax)
plt.show()


# **Figure 2.** Seasonal average (1980-2014) isoprene concentrations UKESM. **DJF**= December, January, February; **MAM** = March, April, May; **JJA** = June, July, August and  **SON**= September, Oktober, November. 

# ### Plot the model difference as % using (NorESM-UKESM)/UKESM*100
# 

# In[11]:


ds_seas_u = ukesm_isop.groupby('time.season').mean() #Group data by the season and obtain the mean UKESM model
ds_seas_n = noresm_isop.groupby('time.season').mean() #Group data by the season and obtain the mean NorESM model

ds_seas_rat = (ds_seas_n-ds_seas_u)/ds_seas_u*100 #calculate the percent difference NorESM model - UKESM model/UKESMmodel*100
ds_seas_rat = ds_seas_rat.where(ds_seas_u.isop>ds_seas_u.quantile(.75))

fig, axs = sp_map(2,2, projection=ccrs.NorthPolarStereo(), figsize=[10,10] )
lat_lims = [60,90]
season_l = ds_seas['season'].values
print(season_l)
plt_ds = ds_seas_rat
for ax, seas in zip(axs.flatten(),season_l):
    _ds= plt_ds.sel(season=seas)['isop'].where(ds_seas['lat']>lat_lims[0])
    _ds.attrs['units']='%'; _ds.attrs['long_name'] = 'Isoprene'
    _ds.plot(ax=ax, robust=True,transform=ccrs.PlateCarree())
    ax.set_title(seas)
    polarCentral_set_latlim(lat_lims, ax)
    add_map_features(ax)
plt.show()


# **Figure 3.** Percent difference between the models (1980-2014) of isoprene concentrations between the two models. With blue color indicating UKESM model having higher isoprene concentrations than NorESM model, and red color indicating vice vera.**DJF**= December, January, February; **MAM** = March, April, May; **JJA** = June, July, August and  **SON**= September, Oktober, November. 

# ### Percent difference in isoprene concentrations between models: NorESM2-LM and UKESM1-0-LL.

# **Table 1** Percent difference in isoprene concentrations between models: NorESM2-LM and UKESM1-0-LL.

# In[12]:


#means of Isoprene [mol/mol] for each of the models
lab_iso = 'Isoprene [mol/mol]'
means=pd.DataFrame(index=['NorESM', 'UKESM', 'UKESM-NorESM %'], columns=['Isoprene [mol/mol]'])
means.loc['NorESM', lab_iso] = 7.8192895e-12
means.loc['UKESM', lab_iso] = 1.13133856e-10

means.loc['UKESM-NorESM %', :]= (means.loc['UKESM', :]- means.loc['NorESM', :])/means.loc['NorESM',:]*100
means


# # EBAS

# ### Load EBAS station data isoprene concentrations:

# In[13]:


# Load station data from EBAS :
# f-tion load_station_data first selects the the station name, then it selectts variable of the interest 'isoprene',
#Convert dates to datetime objects.
def load_station_data(station_name):
    d = open_dod_dic[station_name]
    _d = d['isoprene']
    data = _d.isoprene.data
    tim_dmps = num2date(_d.time.data,units='days since 1900-01-01 00:00:00', calendar='gregorian')
    return pd.DataFrame(data, index=tim_dmps).copy()
#Load Birkenes II data
open_dod_dic={}
open_dod_dic['Birkenes II']=open_dods(
'http://dev-ebas-pydap.nilu.no/' 
'NO0002R.online_ptr.IMG.air.isoprene'
'.1h.NO01L_IONICON_PTR_TOF_8000_BIR.NO01L_internal_proton_transfer_reaction..dods')

Birkenes2_station =load_station_data('Birkenes II') 
##Load Birkenes data
open_dod_dic['Birkenes']=open_dods(
'http://dev-ebas-pydap.nilu.no/' 
'NO0001R.steel_canister.IMG.air.isoprene'
'.1d.NO01L_sc_no1.NO01L_GC_FID_Chrompack_VOCAIR_Analyzer.2.dods')
Birkenes1_station = load_station_data('Birkenes')
#Load Zeppelin station data
open_dod_dic['Zeppelin'] = open_dods(
'http://dev-ebas-pydap.nilu.no/' 
'NO0042G.steel_canister.IMG.air.isoprene'
'.1d.NO01L_sc_no42.NO01L_GC_FID_Chrompack_VOCAIR_Analyzer.2.dods')
Zeppelin_station = load_station_data('Zeppelin')


# ### Compare observation data to models:

# In[14]:


# This is a function from 'negi-stuff' folder, that is getting the station coordinates
#it takes the variable of interest 'isop' (concentration of isoprene), using function 'nearest'. The data are grouped by month.
sys.path.append('negi-stuff')
from negi_stuff.modules.ebas_db_get_info import get_station_coords

def plot_model_ds_station(lat, lon, mod, label, ax=None, var='isop'):
    _du_is = mod.sel(lat=lat, lon=lon, method='nearest')
    _du_is.groupby('time.month').mean()[var].to_series().plot(label=label, ax=ax)
def plot_station_data(df, label, ax=None):
    b=df.copy()
    _ds1 = df.groupby(b.index.month).mean()
    ax.plot(_ds1.index, _ds1,label=label)


# In[51]:


#Plot the standard deviation as shading in python. 
median = ukesm_isop.groupby('time.month').median()
std = ukesm_isop.groupby('time.month').std()
median.sel(lat=slat, lon=slon, method='nearest')
std.sel(lat=slat, lon=slon,method='nearest')
median = median.sel(lat=slat, lon=slon, method='nearest')


# In[273]:


median = median.sel(lat=slat, lon=slon, method='nearest')
std = std.sel(lat=slat, lon=slon,method='nearest')


# In[65]:


#Get the median and SD for plotting error for UKESM model ()
_m = ukesm_isop.groupby('time.month').median()
_s = ukesm_isop.groupby('time.month').std()
median = _m.sel(lat=slat, lon=slon, method='nearest')
std = _s.sel(lat=slat, lon=slon,method='nearest')
#Get the median and SD for plotting error for NorESM model
_m1 =noresm_isop.groupby('time.month').median()
_s1 = noresm_isop.groupby('time.month').std()
median1 = _m1.sel(lat=slat, lon=slon, method='nearest')
std1 = _s1.sel(lat=slat, lon=slon,method='nearest')
##### the above function is supposed to work as I did my best following Sara's guidelines. But,sa sadly, it did not appear
#### on the graph :(

fig, axs = plt.subplots(2, sharex=True, figsize=[15,10])
## Plot observation data of isoprene from Birkenes station and model data.
station_name, slon, slat =get_station_coords('Birkenes II')
plot_model_ds_station(slat, slon, 1e12*ukesm_isop,'Isoprene, pmol/mol, UKESM', ax=axs[0])
plot_model_ds_station(slat, slon, 1e12*noresm_isop,'Isoprene, pmol/mol, NorESM', ax=axs[0])

plot_station_data(Birkenes1_station, 'Isoprene, pmol/mol, Birkenes', ax=axs[0])
plot_station_data(Birkenes2_station, 'Isoprene, pmol/mol, Birkenes II', ax=axs[0])
axs[0].set_title('Station: Birkenes')
ax.fill_between(median['month'], median['isop']-std['isop'], median['isop']+ std['isop'], alpha=0.3, facecolor='g')
ax.fill_between(median1['month'], median1['isop']-std['isop'], median1['isop']+ std1['isop'], alpha=0.3, facecolor='g')

## Plot observation data of Isoprene from Zeppelin mountain station and model data.

station_name, slon, slat =get_station_coords('Zeppelin', unique=False)
plot_model_ds_station(slat, slon, 1e12*ukesm_isop,'Isoprene, pmol/mol, UKESM', ax=axs[1])
plot_model_ds_station(slat, slon, 1e12*noresm_isop,'Isoprene, pmol/mol, NorESM', ax=axs[1])
plot_station_data(Zeppelin_station, 'Isoprene, pmol/mol, Zeppelin', ax=axs[1])
axs[1].set_title('Station: Zeppelin')

for ax in axs:
    ax.set_ylabel('pmol/mol')
    ax.legend()


# **Figure 4** Monthly variations of average (1980-2014) grid-scale isoprene(pmol mol-1) both for the models (NorESM and UKESM) and data from the observations. Upper graph: observation and model averages from Birkenes and Birkenes II stations; Bottom graph: observation and model averages from Zeppelin mountain (Ny-Ålesund). Model data are taken using 'nearest' function in that way obtaining the nearest available value to the observation collection site. 

# # Supplementary information

# ### Get CMIP6 data from NorESM and UKESM models

# #### Isoprene Volume Mixing Ratio  (mol mol-1)

# In[18]:


#This function selects the isoprence concentrations from NorESM2-LM. Due to computing power, the selected years are from
#1980-2015. The region of interest is Arctic, therefore we slice that data by latitudes (60:90). Due to high
#reactivity of the BVOCsThe level of interest is level=0. 
path = '/home/bd15084e-2d25b7-2d47db-2dade3-2daea695ce03d8/shared-cmip6-for-ns1000k/historical/NorESM2-LM/r1i1p1f1/'
var = 'isop'
from_y = '1980-01-01'
to_y = '2015-01-01'
files = glob.glob(path+var+'*')
files.sort()
noresm_isoprene = xr.open_mfdataset(files, combine='nested', concat_dim = 'time'
                          ).sel(time=slice(from_y, to_y))
noresm_isoprene=noresm_isoprene.sel(lat=slice(60,90)).isel(lev=0)
noresm_isoprene.to_netcdf('noresm_isoprenelev0.nc')
noresm_isoprene.close()


# In[ ]:


#This function selects the isoprence concentrations from UKESM1-0-LL. Due to computing power, the selected years are from
#1980-2015. The region of interest is Arctic, therefore we slice that data by latitudes (60:90). Due to high
#reactivity of the BVOCsThe level of interest is level=0.
path = '/home/bd15084e-2d25b7-2d47db-2dade3-2daea695ce03d8/shared-cmip6-for-ns1000k/historical/UKESM1-0-LL/r1i1p1f1/'
var = 'isop'
from_y = '1980-01-01'
to_y = '2015-01-01'
files = glob.glob(path+var+'*')
files.sort()
ukesm_isoprene = xr.open_mfdataset(files, combine='nested', concat_dim = 'time'
                          ).sel(time=slice(from_y, to_y))
ukesm_isoprene=ukesm_isoprene.sel(lat=slice(60,90)).isel(lev=0)
ukesm_isoprene.to_netcdf('ukesm_isoprenelev0.nc')
ukesm_isoprene.close()


# ### Total Emission Rate of Isoprene (kg m-2 s-1)

# In[ ]:


#This function selects the Total Emission Rate of Isoprene (kg m-2 s-1) from NorESM2-LM. Due to computing power, the slected years are from
#1980-2015. The region of interest is Arctic, therefore we slice that data by latitudes (60:90). Due to high
#reactivity of the BVOCsThe level of interest is level=0. 
path = '/home/bd15084e-2d25b7-2d47db-2dade3-2daea695ce03d8/shared-cmip6-for-ns1000k/historical/NorESM2-LM/r1i1p1f1/'
var = 'emiisop'
from_y = '1980-01-01'
to_y = '2015-01-01'
files = glob.glob(path+var+'*')
files.sort()

noresm_is = xr.open_mfdataset(files, combine='nested', concat_dim = 'time'
                          ).sel(time=slice(from_y, to_y))
noresm_isop=noresm_is.sel(lat=slice(60,90)).isel(lev=0)

noresm_isop.to_netcdf('noresm_emis1.nc')
noresm_isop.close()


# In[ ]:


#This function selects the Total Emission Rate of Isoprene (kg m-2 s-1) from UKESM1-0-LL. Due to computing power, the slected years are from
#1980-2015. The region of interest is Arctic, therefore we slice that data by latitudes (60:90). Due to high
#reactivity of the BVOCsThe level of interest is level=0. 
var = 'emiisop'
from_y = '1980-01-01'
to_y = '2015-01-01'
ukesm = xr.open_mfdataset(files, combine='nested', concat_dim = 'time'
                          ).sel(time=slice(from_y, to_y))
path = '/home/bd15084e-2d25b7-2d47db-2dade3-2daea695ce03d8/shared-cmip6-for-ns1000k/historical/UKESM1-0-LL/r1i1p1f2/'
files = glob.glob(path+var+'*')#" get_fl('emibvoc',path )
files.sort()
ukesm_is = xr.open_mfdataset(files, combine='nested', concat_dim = 'time'
                         ).sel(time=slice(from_y, to_y))
ukesm_isop=ukesm_is.sel(lat=slice(60,90)).isel(lev=0)
ukesm_isop.close()


# ### *Pyaerocom* for collocation of the data and obtaining statistics for model comparison

# In[19]:


#Select the NorESM model data
CMIP6_TEST_DIR = '/home/bd15084e-2d25b7-2d47db-2dade3-2daea695ce03d8/'
CMIP6_TEST_FILE = 'noresm_emis1.nc'
path = CMIP6_TEST_DIR + CMIP6_TEST_FILE
modeldata_noresm_conc = pya.GriddedData(path, var_name='emiisop')


# In[71]:


#Select the UKESM model data
CMIP6_TEST_DIR = '/home/bd15084e-2d25b7-2d47db-2dade3-2daea695ce03d8/'
CMIP6_TEST_FILE = 'ukesm_emiisop_4.nc'
path = CMIP6_TEST_DIR + CMIP6_TEST_FILE
modeldata_ukesm_conc = pya.GriddedData(path, var_name='emiisop')


# In[72]:


modeldata_noresm_conc.metadata['ts_type'] = 'monthly' # models ts_type monthly, will be used to collocate data
modeldata_noresm_conc.ts_type


# In[73]:


modeldata_ukesm_conc.metadata['ts_type'] = 'monthly' # models ts_type monthly, will be used to collocate data
modeldata_ukesm_conc.ts_type


# In[74]:


modeldata_noresm_conc.metadata['data_id']='NorESM' #give names to axis
modeldata_ukesm_conc.metadata['data_id']='UKESM' #give names to axis


# In[75]:


modeldata_noresm_conc.start,modeldata_noresm_conc.stop


# In[76]:


modeldata_ukesm_conc.start,modeldata_ukesm_conc.stop


# In[77]:


#collocate the both model data:
try:
    coldata1 = pya.colocation.colocate_gridded_gridded(modeldata_noresm_conc, 
                                                      modeldata_ukesm_conc,
                                                      start=1850,
                                                      ts_type='monthly')
    stats = coldata1.calc_statistics()
except Exception as e:
    print('Colocating failed. Reason: {}'.format(repr(e)))  


# In[78]:


#set x and y axis as min-max from the models
_n= coldata1.data.sel(data_source='NorESM')
_u= coldata1.data.sel(data_source='UKESM')
xlim=[np.nanmin(_n), np.nanmax(_n)]
ylim=[np.nanmin(_u), np.nanmax(_u)]


# In[79]:


#plot the covariance of the two models
coldata1.plot_scatter(marker='o', color='blue', alpha=0.1)#, xlabel='blabla');
plt.xlim(xlim)
plt.ylim(ylim)


# **Figure 5** The relationship between collocated NorESM and UKESM models for Total Emission Rate of Isoprene (kg m-2 s-1). 
