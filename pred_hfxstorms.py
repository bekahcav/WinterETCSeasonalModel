'''
2021 Feb 18
	Rebekah Cavanagh, Dalhousie University
	- Select the storms that pass within x km of Halifax
	- Add the concurrent HFX weather info to their dictionaries
	- Save for future analysis
'''
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sys import path
path.append('/data/home/rebekahc/bek/programs/rc/MSc/tracks_evaluation/')
import eval_functions as ev
path.append('/data/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/')
import pred_functions as pr
import cartopy.crs as ccrs

hfx_obs_hr = pd.read_pickle('~/bek/data/hfx_obs_hourly_compiled_1979-2019.pkl')
hfx_obs_hr['Weather'][hfx_obs_hr['Weather'].isnull()] = 'Missing'

radius_wprecip = '750w1000'
radius, precip_radius = radius_wprecip.split('w')

draw_lon,draw_lat = ev.circ(int(radius),[(360-63.5752),44.6488],50)
hfx_r = ev.makePolygon(draw_lon, draw_lat)
draw_lon_outer,draw_lat_outer = ev.circ(int(precip_radius),[(360-63.5752),44.6488],50)
hfx_r_outer = ev.makePolygon(draw_lon_outer, draw_lat_outer)

yearStart = 1979
yearEnd = 2019
storms_hfx_r = []
tot = 0

for yr in range(yearStart, yearEnd):
	yr_ind = yr - 1979
	# load storms
	storms_full = np.load('/data/home/rebekahc/bek/programs/ecjo/stormTracking-ERA5/storm_track/Met/1deg/tr1000_min_ecc90/interp/storm_track_slp'+str(yr)+'-'+str(yr+1)+'.npz', allow_pickle = True)['storms']
	# choose storms in the appropriate time frame 
	yrstorms_hfx = []
	for ed in storms_full:
		if ((ed['month'][0] > 10) + (ed['month'][-1] < 4)):
			pts_incirc = []
			pts_inoutercirc = []			
			for i in range(len(ed['lon'])):
				pts_incirc.append(ev.inPolygon(hfx_r,(ed['lon'][i],ed['lat'][i])))
				pts_inoutercirc.append(ev.inPolygon(hfx_r_outer,(ed['lon'][i],ed['lat'][i])))
			# if its in hfx area, add weather keys to the storm dict, add the storm to the hfx storms list
			if np.any(pts_incirc):
				ed['within_hfx'] = np.array(pts_incirc)
				ed['near_hfx'] = np.array(pts_inoutercirc)
				ed['weather_hfx'] = pr.get_observation(hfx_obs_hr, 'Weather', ed)
				ed['accum_precip_hfx'] = None
				ed['wind_hfx'] = {'dir': pr.get_observation(hfx_obs_hr, 'Wind Dir (10s deg)', ed), 'speed': pr.get_observation(hfx_obs_hr, 'Wind Spd (km/h)', ed)}
				ed['storm_type'] = pr.get_storm_type(ed['weather_hfx'], ed['near_hfx'])
				print(ed['storm_type'])

				yrstorms_hfx.append(ed)
				tot+=1

	storms_hfx_r.append(yrstorms_hfx)

# save the Halifax storms dictionary 
storms_hfx_r = np.array(storms_hfx_r, dtype=object)
var = 'storms_hfx_r'+radius+'_state'
np.savez('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_predictands/storms_hfx_r'+radius+'_state.npz',**{var:storms_hfx_r})

#####################################################################
#			 separate into timeseries by precip state 				#
#																	#
# no precip: less than 3 hours of precipitaion						#
# precip: at least 3 hrs precip (liquid + freezing + frozen)		#
# liquid: at least 90% of precip timesteps with 'rain' or 'drizzle'	#
# frozen: at least 90% of precip ts with 'snow' or 'ice pellets'	#
# mixed: not dominantly snow or rain - 
# windy: at least one timestep with wind > 44 km/h (98th percentile)#
#####################################################################

storms_hfx_state = np.load('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_predictands/storms_hfx_r'+radius+'_state.npz',allow_pickle=True)['storms_hfx_r'+radius+'_state']

subseries = {'total':np.zeros(len(range(yearStart,yearEnd))),'total_precip':np.zeros(len(range(yearStart,yearEnd))),'snow_count':np.zeros(len(range(yearStart,yearEnd))),'rain_count':np.zeros(len(range(yearStart,yearEnd))),'mixed_count':np.zeros(len(range(yearStart,yearEnd))),'no_precip_count':np.zeros(len(range(yearStart,yearEnd))),'wind_count':np.zeros(len(range(yearStart,yearEnd)))}
substorms = {'snow_sts' : [],'rain_sts' : [],'mixed_sts' : [],'no_precip_sts' : [],'wind_sts' : []}

# calculate the subseries
for y in range(len(storms_hfx_state)):
	rain_yr_sts = []
	snow_yr_sts = []
	mixed_yr_sts = []
	no_precip_yr_sts = []
	wind_yr_sts = []

	storms_hfx_year = storms_hfx_state[y]
	for i in range(len(storms_hfx_year)):
		st = storms_hfx_year[i]
		states = pr.categorize_precipstate(st['weather_hfx'],st['near_hfx'])
		hrs_precip = (states['Liquid']+states['Freezing']+states['Frozen'])
		winds = st['wind_hfx']['speed'][st['near_hfx']]
		if hrs_precip > 2 : 
			if (st['storm_type'][0] == "Frozen") and (states['Frozen'] > 2) and ((states['Frozen']/hrs_precip) > .9):
					subseries['snow_count'][y]+=1
					snow_yr_sts.append(st)
			elif (st['storm_type'][0] == "Liquid") and (states['Liquid'] > 2) and ((states['Liquid']/hrs_precip) > .9):
					subseries['rain_count'][y]+=1
					rain_yr_sts.append(st)
			else:
					subseries['mixed_count'][y]+=1
					mixed_yr_sts.append(st)
			subseries['total_precip'][y]+=1
		else :
			subseries['no_precip_count'][y]+=1
			no_precip_yr_sts.append(st)

		if np.any(winds > 44):  
			subseries['wind_count'][y]+=1
			wind_yr_sts.append(st)

		subseries['total'][y]+=1

	substorms['rain_sts'].append(rain_yr_sts)
	substorms['snow_sts'].append(snow_yr_sts)
	substorms['mixed_sts'].append(mixed_yr_sts)
	substorms['no_precip_sts'].append(no_precip_yr_sts)
	substorms['wind_sts'].append(wind_yr_sts)

np.savez('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_predictands/series_hfx_r'+radius+'_state.npz',subseries=subseries, substorms=substorms)
