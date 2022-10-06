'''
  Software for the tracking of storms and high-pressure systems
  Written by Dr. Eric Oliver for NCEP20CR
  adapted to ERA5 dataset by Rebekah Cavanagh
'''

#
# Load required modules
#

import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from sys import path
path.append('/data/home/rebekahc/bek/programs/ecjo/stormTracking-ERA5/')
import storm_functions as storm
import datetime as dtm

#
# Load in slp data and lat/lon coordinates
#

# Parameters
pathroot = '/home/oliver/data/ERA/ERA5/MSLP/'

# Generate date and hour vectors
yearStart = 1979
yearEnd = 2019

#cut each season (oct - apr)
for yr in range(yearStart,yearEnd):
    ti, datesi, Ti, yeari, monthi, dayi, doyi = storm.timevector([yr,10,1], [yr+1,4,30]) # Daily
    if yr == yearStart:
        t = ti
        dates = datesi
        T = Ti
        year = yeari
        month = monthi
        day = dayi
        doy = doyi
    else:
        t = np.append(t,ti)
        dates = np.append(dates,datesi)
        T += Ti
        year = np.append(year,yeari)
        month = np.append(month,monthi)
        day = np.append(day,dayi)
        doy = np.append(doy,doyi)
#choose timestep
dt = 3
#repeat every day's value for the number of timesteps in the day
year = np.repeat(year, 24/dt) 
month = np.repeat(month, 24/dt) 
day = np.repeat(day, 24/dt) 
hour = np.tile(np.arange(0,24,dt), T) 

# Load lat, lon
filename = pathroot + 'ERA5_MSLP_' + str(yearStart) + '.nc'
fileobj = Dataset(filename, 'r')
lon = fileobj.variables['longitude'][:].astype(float)
lat = fileobj.variables['latitude'][:].astype(float)
fileobj.close()

#chose the area of interest
lon1 = np.where(lon==250)[0][0]
lon2 = np.where(lon==359.75)[0][0]
lat1 = np.where(lat==70)[0][0]
lat2 = np.where(lat==25)[0][0]
cut_lon = [1,2,4] #smoothing threshold 
cut_lat = [1,2,4] #smoothing threshold


# Load slp data
for yr in range(yearStart, yearEnd+1):
    for i in ['oct','apr']:
        
        #set the start and end timestep
        filename = pathroot + 'ERA5_MSLP_' + str(yr) + '.nc'
        fileobj = Dataset(filename, 'r')
        #the ordinal time as defined by the datetime module
        time = storm.todtm(fileobj.variables['time'][:].astype(float))
        oc = np.where(dtm.date(yr,10,1).toordinal()==time)[0][0]
        ap = np.where(dtm.date(yr,4,30).toordinal()==time)[0][0]
        if i =='oct':        
            slp = fileobj.variables['msl'][oc:,lat1:lat2,lon1:lon2].astype(float)
        if i =='apr':        
            slp = fileobj.variables['msl'][:ap,lat1:lat2,lon1:lon2].astype(float)
        fileobj.close()

        for deg in cut_lon:
            #
            # Storm Detection
            #

            # Initialisation

            lon_storms_c = []
            lat_storms_c = []
            amp_storms_c = []
            area_storms_c = []
            # Loop over time

            T = slp.shape[0]

            for tt in np.arange(0, T, 3):
                    #
                    # Detect lon and lat coordinates of storms at each time step
                    #
                    lon_storms, lat_storms, amp, area, field = storm.detect_storms_Met(slp[tt,:,:], lon[lon1:lon2], lat[lat1:lat2], res=.25, order='topdown' , Npix_min=9, Npix_max= 6000, rel_amp_thresh=100, d_thresh=2500, cyc='cyclonic', cut_lon=deg, cut_lat=deg, globe=False)
                    lon_storms_c.append(lon_storms)
                    lat_storms_c.append(lat_storms)
                    amp_storms_c.append(amp)
                    area_storms_c.append(area)
                    #
                    # Save as we go (every 100 time steps and at the end of the year)
                    #
                    if (np.mod(tt, 100) == 0) + (tt == T-1):
                        print ('Save data... ','step ', tt/3, ' of ', T/3)
                    #
                    # Combine storm information from all days in the season into a list, and save
                    #
                        storms = storm.storms_list_c(lon_storms_c, lat_storms_c, amp_storms_c, area_storms_c)
                        print('length of storm list: ',len(storms)) 
                        np.savez('/home/rebekahc/bek/programs/ecjo/stormTracking-ERA5/storm_det/Met/'+str(deg)+'deg/storm_det_slp-'+str(deg)+'deg_'+str(yr)+str(i), storms=storms)
            #
            #Save at the end of the loop over time 
            #
            print ('Save data...',  tt, T)
            storms = storm.storms_list_c(lon_storms_c, lat_storms_c, amp_storms_c, area_storms_c)            
            np.savez('/home/rebekahc/bek/programs/ecjo/stormTracking-ERA5/storm_det/Met/'+str(deg)+'deg/storm_det_slp-'+str(deg)+'deg_'+str(yr)+str(i),storms=storms)
            print (' deg:',deg,' szn:',i, ' year:',yr)
            print ('done deg:',deg)
        print ('done szn:',i)
    print('done year:',yr)
