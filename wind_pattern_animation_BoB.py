#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount("/content/drive/")


# In[ ]:


get_ipython().system('pip install gsw==3.0.1')

get_ipython().system('pip install "basemap == 1.3.0b1" "basemap-data == 1.3.0b1"')

get_ipython().system('pip install basemap')

get_ipython().system('pip install cartopy')

get_ipython().system('pip install shapely --no-binary shapely cartopy')

get_ipython().system('pip install --no-binary shapely shapely --force')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xr

from mpl_toolkits.basemap import Basemap

import matplotlib as mpl

import os
from datetime import datetime
from matplotlib import pyplot as plt




import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker


# In[ ]:


parent = "drive/MyDrive/wind_data/Wind_data_monthly/"
path = [parent+i for i in os.listdir(parent) if i!= "output" and i!= "wind_pressure_data"]
path


# In[ ]:


data= xr.open_mfdataset(path[0:3])
data


# In[ ]:


u10_ = data.u10.values

v10_ = data.v10.values

#u10_1 = u10_[:2]
u10  = np.array([(sum(u10_[i:i+3])/3, sum(u10_[i+3:i+7])/4, u10_[i+7], sum(u10_[i+8:i+12])/4)  for i in range(2, 228, 12)])

v10  = np.array([(sum(v10_[i:i+3])/3, sum(v10_[i+3:i+7])/4, v10_[i+7], sum(v10_[i+8:i+12])/4)  for i in range(2, 228, 12)])




lat = data.latitude.values

lon = data.longitude.values

#lat, lon

max(lon), min(lon), max(lat), min(lat), u10.shape


# In[ ]:


parent2 = "drive/MyDrive/wind_data/Wind_data_monthly/wind_pressure_data/"
path2 = [parent2+i for i in os.listdir(parent2)][0]


data2 = xr.open_mfdataset(path2)

data2


# In[ ]:



pres_ = data2.sp.values[:, :, :]

pres = np.array([(sum(pres_[i:i+3, :, :])/3, sum(pres_[i+3:i+7, :, :])/4, pres_[i+7, :, :], 
                  sum(pres_[i+8:i+12, :, :])/4)  for i in range(2, 228, 12)])

pres.shape


# In[ ]:


parent3 = "drive/MyDrive/wind_data/Wind_data_monthly/SST_data_2003-2021/"
path3 = [parent3+i for i in os.listdir(parent3)][0]


data3 = xr.open_mfdataset(path3)

data3


# In[ ]:



sst_ = data3.sst.values[:, :, :]

sst = np.array([(sum(sst_[i:i+3, :, :])/3, sum(sst_[i+3:i+7, :, :])/4, sst_[i+7, :, :], 
                  sum(sst_[i+8:i+12, :, :])/4)  for i in range(2, 228, 12)])

sst.shape


# In[ ]:


def plot(lat, lon, u, v, variable, variable_name, output_folder):

  x, y = np.meshgrid(lon, lat)

  year = [2003+i for i in range(19)]

  seasons = ["Spring", "Summer", "Autumn", "Winter"]

  
  for j in range(len(u)):

    for i in range(4):
      
      fig = plt.figure(figsize=(15, 8))

      ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
      ax.set_extent([max(lon), min(lon), max(lat), min(lat)], ccrs.PlateCarree()) 
 


      ax.add_feature(cfeature.LAND, color = 'gray', lw = 2, linestyle = "-")
      ax.add_feature(cfeature.COASTLINE, color = 'k',lw =2,  linestyle = "-")
      ax.add_feature(cfeature.BORDERS,color = 'k', lw =2, linestyle='-')
      ax.add_feature(cfeature.RIVERS, color = 'k',lw =2, linestyle=':')


      x1 = (max(lon)-min(lon))/4
      y1 = (max(lat)-min(lat))/4

      ax.set_xticks([min(lon),min(lon)+x1, min(lon)+2*x1, min(lon)+3*x1, max(lon)])
      ax.set_yticks([min(lat),min(lat)+y1, min(lat)+2*y1, min(lat)+3*y1,max(lat)])

      ax.set_yticklabels([])
      ax.set_xticklabels([])

      gl = ax.gridlines()
      gl.bottom_labels = True
      gl.left_labels = True
      gl.xlines = False
      gl.ylines = False

      gl.xlocator = mticker.FixedLocator(np.arange(int(min(lon)),int(max(lon)+1),10))
      gl.ylocator = mticker.FixedLocator(np.arange(int(min(lat)), int(max(lat)+1),5))
      gl.xlabel_style = {'size': 11, 'color': 'gray', 'fontweight': 'bold'}
      gl.ylabel_style = {'size': 11, 'color': 'gray', 'fontweight': 'bold'}

      if variable_name =="SST":
        vmin_, vmax_, scale = 284, 306, 20
      else:
        vmin_, vmax_, scale = np.nanmax(variable)-5000, np.nanmax(variable), 20
      m = ax.contourf(lon, lat, variable[j][i], cmap = 'hsv',
                      vmin = vmin_,vmax =  vmax_,
                      levels = np.linspace(vmin_, vmax_, scale))
    
     

      plt.colorbar(m)

      
      plt.streamplot(x,y, u[j][i], v[j][i], color = 'white', density = 2)


      plt.title("Wind vector {v} year {y} season {s}\n".format(v = variable_name, y = year[j],s = seasons[i]), fontweight = 'bold')
      plt.savefig(output_folder+"year_{y}_season_{s}_wind_vector_{v}.png".format(y = year[j],s = seasons[i], v = variable_name))
     
      plt.show()


    
    


# In[ ]:


output_folder = parent+"output/"
if not os.path.exists(output_folder): os.mkdir(output_folder)

plot(lat, lon,u10, v10, pres, "pressure", output_folder)


# In[ ]:


plot(lat, lon,u10, v10, sst, "SST", output_folder)


# In[138]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


year = [2003+i for i in range(19)]

seasons = ["Spring", "Summer", "Autumn", "Winter"]

print("Select either SST or pressure")

variable_name = input()

fig, ax = plt.subplots()
ims = []
for j in range(len(u10)):
  
  for i in range(4):
    
    m = plt.imread(output_folder+"year_{y}_season_{s}_wind_vector_{v}.png".format(y = year[j],s = seasons[i], v = variable_name))
    
    im = ax.imshow(m, animated=True)

    ax = plt.gca()

    #hide x-axis
    ax.get_xaxis().set_visible(False)

    #hide y-axis 
    ax.get_yaxis().set_visible(False)

    
    
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                repeat_delay=1000)


ax = plt.gca()

#hide x-axis
ax.get_xaxis().set_visible(False)

#hide y-axis 
ax.get_yaxis().set_visible(False)

dpi = 300
writer = animation.writers['ffmpeg'](fps=2)
ani.save(output_folder+"movie_wind_{}.mp4".format(variable_name),writer=writer,dpi=dpi)


# In[148]:




springP = sum([i[0] for i in pres])/19
summerP = sum([i[1] for i in pres])/19
autumnP = sum([i[2] for i in pres])/19
winterP = sum([i[3] for i in pres])/19

sesP = [springP, summerP, autumnP, winterP]

springT = sum([i[0] for i in sst])/19
summerT = sum([i[1] for i in sst])/19
autumnT = sum([i[2] for i in sst])/19
winterT = sum([i[3] for i in sst])/19

sesT = [springT, summerT, autumnT, winterT]


springU = sum([i[0] for i in u10])/19
summerU = sum([i[1] for i in u10])/19
autumnU = sum([i[2] for i in u10])/19
winterU = sum([i[3] for i in u10])/19

sesU = [springU, summerU, autumnU, winterU]



springV = sum([i[0] for i in v10])/19
summerV = sum([i[1] for i in v10])/19
autumnV = sum([i[2] for i in v10])/19
winterV = sum([i[3] for i in v10])/19

sesV = [springV, summerV, autumnV, winterV]


# In[178]:



from matplotlib import gridspec

plt.style.use('ggplot')
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True



def seasonal_plot(data, variable_name, u = u10, v = v10, lon = lon,lat = lat, colormap = "RdBu_r", scale = 20, output = output_folder):
  
  x, y = np.meshgrid(lon, lat)
  nrow = 2
  ncol = 2
  
  seasons = ["Spring", "Summer", "Autumn", "Winter"]
  fig = plt.figure(figsize=((ncol+1)*4, (nrow+1)*4))

  

  gs = gridspec.GridSpec(nrow, ncol,wspace=0.0, hspace=0.0, top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), left=0.07, right=0.93) 
  
  j = 0
  for i in range(nrow):
    for k in range(ncol):
      
      ax = plt.subplot(gs[i,k], projection=ccrs.PlateCarree())
    
      extent = [30,120,-40,30]
      ax.set_extent(extent, ccrs.PlateCarree()) 


      ax.add_feature(cfeature.LAND, color = 'gray', lw = 2, linestyle = "-")
      ax.add_feature(cfeature.COASTLINE, color = 'k',lw =2,  linestyle = "-")
      ax.add_feature(cfeature.BORDERS,color = 'k', lw =2, linestyle='-')
      ax.add_feature(cfeature.RIVERS, color = 'k',lw =2, linestyle=':')




      ax.set_xticks([30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
      ax.set_yticks([-40, -30,-20, -10,0, 10, 20, 30])

      ax.set_yticklabels([])
      ax.set_xticklabels([])

      gl = ax.gridlines()

      if i==0 and k==0:
        gl.top_labels = True
        gl.bottom_labels = False
        gl.left_labels = True
        gl.xlines = False
        gl.ylines = False

        gl.xlocator = mticker.FixedLocator(np.arange(30,120,10))
        gl.ylocator = mticker.FixedLocator(np.arange(-40,30,10))

        gl.xlabel_style = {'size': 10, 'color': 'k', 'fontweight':'bold'}
        gl.ylabel_style = {'size': 10, 'color': 'k', 'fontweight':'bold'}



  


        if variable_name =="SST":
          vmin_, vmax_, scale = 284, 306, 20
        else:
          vmin_, vmax_, scale = np.nanmax(data)-5000, np.nanmax(data), 20
        
        m = ax.contourf(lon, lat, data[j],cmap = colormap, transform = ccrs.PlateCarree(),extent = extent,
                      vmin = vmin_,vmax =  vmax_,
                      levels = np.linspace(vmin_, vmax_, scale))
    
     

       

      
        plt.streamplot(x,y, sesU[j], sesV[j], color = 'white', density = 2)
        
    
    
        ax.text(67,26,"{}".format(seasons[j]), bbox = {"facecolor":"white"}, fontweight = 'bold', fontsize = 15)
        
        j+=1

      else:
        gl.top_labels = False
        gl.bottom_labels = False
        gl.left_labels = False
        gl.xlines = False
        gl.ylines = False

        
    


        if variable_name =="SST":
          vmin_, vmax_, scale = 284, 306, 20
        else:
          vmin_, vmax_, scale = np.nanmax(data)-5000, np.nanmax(data), 20
        
        m = ax.contourf(lon, lat, data[j],cmap = colormap, transform = ccrs.PlateCarree(),extent = extent,
                      vmin = vmin_,vmax =  vmax_,
                      levels = np.linspace(vmin_, vmax_, scale))
    
     

        

      
        plt.streamplot(x,y, u[j][i], v[j][i], color = 'white', density = 2)
    
        ax.text(67,26,"{}".format(seasons[j]), bbox = {"facecolor":"white"}, fontweight = 'bold', fontsize = 15)
        j+=1



  plt.suptitle("Seasonal Wind vector {v}\n".format(v = variable_name), fontweight = 'bold', fontsize = 18)
  cbar_ax = fig.add_axes([0.9, 0.1, 0.04, 0.8])
  fig.colorbar(m,  cax=cbar_ax)
  
  plt.savefig(output+"seasonal_wind_vector_{v}.png".format(v = variable_name))

  plt.show()

   


# In[179]:


seasonal_plot(sesP, "Pressure")


# In[180]:


seasonal_plot(sesT, "SST")
