'''

  Algorithm for the tracking of storms
  based on detected storm position data.

  run after storm_detection.py

'''

# Load required modules

import numpy as np
from netCDF4 import Dataset

#cd /data/home/rebekahc/bek/programs/ecjo/stormTracking-ERA5/

import storm_functions as storm

#
# Automated storm tracking
#

yearStart = 1979
yearEnd = 2018

dt = 3

#setting up month day and hour vectors
lp = [ np.repeat(np.append(np.repeat(10,31), np.append(np.repeat(11,30), np.append(np.repeat(12,31), np.append(np.repeat(1,31), np.append(np.repeat(2,29), np.append(np.repeat(3,31), np.repeat(4,29))))))),24),#month
np.repeat(np.append(np.arange(1,32), np.append(np.arange(1,31), np.append(np.arange(1,32), np.append(np.arange(1,32), np.append(np.arange(1,30), np.append(np.arange(1,32), np.arange(1,30))))))),24) ,#day
np.tile(np.arange(0,24),212)]#hour
# if not hourly data, adjust to the timestep
leap = [lp[0][np.arange(0,len(lp[0]),dt)],lp[1][np.arange(0,len(lp[0]),dt)], lp[2][np.arange(0,len(lp[0]),dt)]]

nn = [ np.repeat(np.append(np.repeat(10,31), np.append(np.repeat(11,30), np.append(np.repeat(12,31), np.append(np.repeat(1,31), np.append(np.repeat(2,28), np.append(np.repeat(3,31), np.repeat(4,29))))))),24),#month
np.repeat(np.append(np.arange(1,32), np.append(np.arange(1,31), np.append(np.arange(1,32), np.append(np.arange(1,32), np.append(np.arange(1,29), np.append(np.arange(1,32), np.arange(1,30))))))),24)  ,#day
np.tile(np.arange(0,24),211)]#hour
# if not hourly data, adjust to the timestep
non = [nn[0][np.arange(0,len(nn[0]),dt)], nn[1][np.arange(0,len(nn[0]),dt)],nn[2][np.arange(0,len(nn[0]),dt)]]

# Load in detected positions and date/hour information
for tr in [1000,1500]:
	for deg in [0,1,2,3,4,5]:
		for yr in range(yearStart, yearEnd+1):
			print(yr)

			f_oct = '/home/rebekahc/bek/programs/ecjo/stormTracking-ERA5/storm_det/'+str(deg)+'deg/storm_det_slp-'+str(deg)+'deg_'+str(yr)+'oct.npz'
			f_apr = '/home/rebekahc/bek/programs/ecjo/stormTracking-ERA5/storm_det/'+str(deg)+'deg/storm_det_slp-'+str(deg)+'deg_'+str(yr+1)+'apr.npz'

			data_oct = np.load(f_oct, allow_pickle=True)
			data_apr = np.load(f_apr, allow_pickle=True)

			#apply terrain filter to remove storms at higher elevation than threshold from the tracking process
			filename = '/home/oliver/data/ERA/ERA5/ORO/ERA5_ORO.nc'
			fileobj = Dataset(filename, 'r')
			terrain = fileobj.variables['z'][0]/9.80665
			lon = fileobj.variables['longitude'][:].astype(float)
			lat = fileobj.variables['latitude'][:].astype(float)		
			fileobj.close()
			det_storms_oct = storm.terrain_filter(data_oct['storms'], terrain, lat, lon, threshold = tr)
			det_storms_apr = storm.terrain_filter(data_apr['storms'], terrain, lat, lon, threshold = tr)

			det_storms = np.append(det_storms_oct, det_storms_apr)
			#each det_storms[i] is a dict of all min at that timestep	

			year = np.append(np.repeat(yr,len(det_storms_oct)),np.repeat(yr+1,len(det_storms_apr)))
			if len(year) == len(leap[0]):
				month, day, hour = leap
			elif len(year) == len(non[0]):
				month, day, hour = non
			# Initialize storms discovered at first time step

			storms = storm.storms_init(det_storms, year, month, day, hour)

			# Stitch storm tracks together at future time steps

			T = len(det_storms) # number of time steps
			for tt in range(1, T):
	#		    print (tt, T)
			    # Track storms from time step tt-1 to tt and update corresponding tracks and/or create new storms
			    storms = storm.track_storms(storms, det_storms, tt, year, month, day, hour, dt=3)

			# Add keys for storm age and flag if storm was still in existence at end of run
			for ed in range(len(storms)):
			    storms[ed]['age'] = len(storms[ed]['lon'])

			# Strip storms based on track lengths
				# dt - 3 hourly data, d_tot_min/d_ratio - no minimum track length, dur_min - must last 12 hrs
			storms = storm.strip_storms(storms, dt=3, d_tot_min=0., d_ratio=0., dur_min=12)

			# Save tracked storm data
			np.savez('./storm_track/'+str(deg)+'deg/terrain_'+str(tr)+'/storm_track_slp'+str(yr)+'-'+str(yr+1), storms=storms) 

