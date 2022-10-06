'''

2021 Feb 18

	Rebekah Cavanagh, Dalhousie University
	Functions to support the prediction module 


'''

import numpy as np 
import pandas as pd 

def get_observation(obs, attribute, storm):
	'''
	selects the specified attribute from the weather data for the timesteps of the specified storm 
	--------------------------------------------------------
	inputs:
		attribute - str - the key from the obs dataframe
		storm - dict - the individual storm dictionary

	returns
		a list of the specified attribute for the timesteps of the given storm

	'''

	start = pd.Timestamp(year=storm['year'][0], month=storm['month'][0], day=storm['day'][0], hour=storm['hour'][0], tz='UTC')
	stop = pd.Timestamp(year=storm['year'][-1], month=storm['month'][-1], day=storm['day'][-1], hour=storm['hour'][-1], tz='UTC')

	att_list = np.array(obs[start:stop][attribute])

	return att_list


def get_storm_type(weather, valid, method="precip_state"):
	'''
	selects the storm precipitation type given the weather remarks	-----------------------------------------------------------------
	input: 
		weather - list of strings - from ECCC observations, the written remarks of the weather at each timestep of the storm's lifetime
		valid - a list of T/F values indicating which weather timesteps to consider (a mask that can indicate e.g. at which timesteps the system is within a region of interest)
		method - str - the way in which the storms will be categorized and what categories will be used

	returns
		the storm type 

	'''
	
	if method == "precip_type":
		cats = categorize_preciptype(weather, valid)
	if method == "precip_state":
		cats = categorize_precipstate(weather, valid)	

	highest_name, highest_val = most_common_type(cats)

	return highest_name, highest_val


def most_common_type(cats_dict):

	highest_name = list(cats_dict.keys())[0]
	highest_val = cats_dict[highest_name]

	for key in list(cats_dict.keys())[1:]:
		if cats_dict[key] > highest_val:
			highest_name = key 
			highest_val = cats_dict[key]
		elif cats_dict[key] == highest_val:
			highest_name += (" and "+key)

	if highest_val == 0:
		highest_name = 'No recorded precipitation'

	return highest_name, highest_val

def categorize_preciptype(weather, valid):
	'''
	categorizes each timestep based on precipitation type using the weather remarks
	-----------------------------------------------------------------
	input: 
		weather - list of strings - from ECCC observations, the written remarks of the weather at each timestep of the storm's lifetime
		valid - a list of T/F values indicating which weather timesteps to consider (a mask that can indicate e.g. at which timesteps the system is within a region of interest)

	returns
		dict{'Rain','Ice', 'Snow'} - the categories and number of timesteps the observed conditions fall in that category; rain and drizzle are classified as rain; ice pellet(s), freezing rain, and freezing drizzle are classified as ice; snow grains, snow, and blowing snow are classified as snow

	'''
	rain = 0
	ice = 0
	snow = 0
	none = 0

	for i in range(len(weather)):
		if valid[i]:
			rmrk = weather[i]
			if ("Rain" in rmrk) or ("Drizzle" in rmrk) and ("Freezing" not in rmrk):
				rain += 1
			if ("Ice" in rmrk) or ("Freezing" in rmrk):
				ice += 1
			if ("Snow" in rmrk):
				snow += 1
			if ( ("Snow" not in rmrk) and ("Ice" not in rmrk) and ("Freezing" not in rmrk) and ("Rain" not in rmrk) and ("Drizzle" not in rmrk) ) :
				none+=1

	if (rain + snow + ice + none) < sum(valid):
		print("Unexpected Error: timestep not categorized")
		raise


	cats = {"Rain":rain, "Ice":ice, "Snow":snow}

	return cats

def categorize_precipstate(weather, valid):
	'''
	categorizes each timestep based on precipitation state using the weather remarks
	-----------------------------------------------------------------
	input: 
		weather - list of strings - from ECCC observations, the written remarks of the weather at each timestep of the storm's lifetime
		valid - a list of T/F values indicating which weather timesteps to consider (a mask that can indicate e.g. at which timesteps the system is within a region of interest)

	returns
		dict{'liquid','freezing', 'frozen'} - the categories and number of timesteps the observed conditions fall in that category; liquid: rain, drizzle. freezing: freezing rain, freezing drizzle, freezing fog. frozen: ice pellets, snow pellets, snow grains, snow, blowing snow

	'''
	liquid = 0
	freezing = 0
	frozen = 0
	none = 0

	for i in range(len(weather)):
		if valid[i]:
			rmrk = weather[i]
			if ("Rain" in rmrk) or ("Drizzle" in rmrk) and ("Freezing" not in rmrk):
				liquid += 1
			if ("Freezing" in rmrk):
				freezing += 1
			if ("Snow" in rmrk) or ("Ice Pellets" in rmrk):
				frozen += 1
			if ( ("Snow" not in rmrk) and ("Ice Pellets" not in rmrk) and ("Freezing" not in rmrk) and ("Rain" not in rmrk) and ("Drizzle" not in rmrk) ) :
				none+=1

	if (liquid + freezing + frozen + none) < sum(valid):
		print("Unexpected Error: timestep not categorized")
		raise

	cats = {"Liquid":liquid, "Freezing":freezing, "Frozen":frozen}

	return cats


def extract_ts(target_loc, filename, varname, timeframe, method='nearest_neighbour'):
	'''
	target_loc (tuple): 
		lat lon of location for timeseries extraction
	filename (.nc file): 
		ERA5 data file 
	varname (str): 
		the name of the variable key for the desired data within fileobj
	timeframe (tuple of strings with format 'YYYY-MM-DD HH:MM'):
		start, stop - first and last time step of time series
	method (str):
		how to extract the data from the grid... one of 'nearest_neighbour' - could add 'bilinear_interp','distance_weighted_mean'

	returns a pandas dataframe of the desired info
	'''

	from netCDF4 import Dataset
	import pandas as pd

	# open the nc file and load the variable of interest
	fileobj = Dataset(filename, 'r')
	lons = fileobj.variables['longitude'][:].astype(float)
	lats = fileobj.variables['latitude'][:].astype(float)

	# first make sure the requested lat and lon are within the extent
	find_lat, find_lon = target_loc
	lat_between = (np.min(lats) <= find_lat <= np.max(lats))
	lon_between = (np.min(lons) <= find_lon <= np.max(lons))
	in_bounds = lon_between*lat_between

	if in_bounds == False:
		print('Location outside of satellite extent')
		return "Skip"

	elif in_bounds == True:

		# then determine the indices of the start and end of the specified time frame 
		time_nc = fileobj.variables['time'][:].astype(int)
		time = pd.to_datetime(time_nc, unit='h', origin=pd.Timestamp('1900-01-01 0:00'))
		start, stop = timeframe
		start = pd.Timestamp(start)
		stop = pd.Timestamp(stop)

		i_start = np.where(time>=start)[0][0]
		i_stop = np.where(time<=stop)[-1][-1]

		# extract the data from the time frame of interest
		data = fileobj.variables[varname][i_start:i_stop].astype(float)
		fileobj.close()

		# extract 
		output = pd.DataFrame(index=time[i_start:i_stop])

		if method=='nearest_neighbour':

			diff_lat = np.abs(lats-find_lat)
			diff_lon = np.abs(lons-find_lon)
			i_loc = np.argmin(diff_lat)  
			j_loc =  np.argmin(diff_lon)

			# extract the data
			output[varname] = data[:,i_loc,j_loc]
			output['lat'] = lats[i_loc]
			output['lon'] = lons[j_loc]
			output['method'] = 'nearest_neighbour'

			return output
 
def corr_plot(df, key1, key2, title, lag=None):
	'''
	run a correlation between two timeseries
	the two timeseries should be the same length
	-----
	adding a lag shifts the timeseries 
	'''
	#set up empty array
	from scipy import signal
	import matplotlib.pyplot as plt
	import numpy as np

	plt.figure()
	if lag == None:
		plt.scatter(df[key1],df[key2],s=2,c='slategrey', alpha=.33)
		valids = df[key1].notnull()&df[key2].notnull()
		cf = np.corrcoef(df[valids][key1],df[valids][key2])
	elif lag > 0 :
		plt.scatter(df[key1][:-lag],df[key2][lag:],s=2,c='steelblue', alpha=.33)
		cf = np.corrcoef(df[df[key1].notnull()][key1][:-lag],df[df[key2].notnull()][key2][lag:])
	elif lag < 0 :
		plt.scatter(df[key1][-lag:],df[key2][:lag],s=2,c='steelblue', alpha=.33)
		cf = np.corrcoef(df[df[key1].notnull()][key1][-lag:],df[df[key2].notnull()][key2][:lag])
	
	plt.xlabel(key1)
	plt.ylabel(key2)
	plt.title(title)

	plt.figtext(.59, 0.28,'r = '+str(cf[0,1]))

	return cf 

def corrcoef_map(array_, ts):
	'''
	takes a time series and a map and returns the correlation coefficient of each grid cell
	
	make them 0 mean then

	R(i,j) = sum( array(i,j) x ts ) / np.sqrt( (var(array(i,j)) x var(ts))


	'''
	# remove mean
	ts_0 = ts - ts.mean()
	array_0 = array_ - array_.mean(axis=-1)[:,:,np.newaxis]

	# make the ts into an array of the same dimensions
	ts_array = np.tile(ts_0, array_0.shape[0]*array_0.shape[1]).reshape(array_0.shape)

	# make an array of the denominator terms
	denom_ts = np.sum((ts_array**2),axis=(2))
	denom_array = np.sum((array_0**2),axis=(2))

	# calculate the denominator of the calc
	denom = np.sqrt(denom_ts*denom_array)

	# calculate the numerator of the calc
	numer = np.sum( (array_0*ts_array) , axis=2)

	# final 
	R = numer/denom

	return(R)

def partial_corrcoef_map(array_, ts, first_corr_ts, max_corr):
	'''
	calculate the partial correlation of the array with the time series accounting for the influence of the first_corr_ts timeseries

	formula:
	R(12.3) = ( R(12)- R(13)R(23) ) / 
			( sqrt(1-R(13)**2)sqrt(1-R(23)**2) )
	'''

	R_12 = corrcoef_map(array_, ts)
	R_13 = corrcoef_map(array_, first_corr_ts)
	R_23 = max_corr

	R_123 = ((R_12 - (R_13*R_23)) / (np.sqrt(1-(R_13**2))*np.sqrt(1-(R_23**2))))

	return(R_123)



def mean_spatial_smooth(X, centre_weight=1):
	'''
	get the average of the 8 surrounding cells and the cell itself
	 - surrounding cells are weighted as 1, weight of centre defaults to 1 but may be changed as a keyword arg

	 resulting array is 2 rows and 2 columns shorter - i.e. the outside ring of the array is gone

	'''

	centre = X[1:-1,1:-1]*centre_weight

	centre += X[:-2,:-2]
	centre += X[1:-1,:-2]
	centre += X[2:,:-2]
	centre += X[2:,1:-1]
	centre += X[2:,2:]
	centre += X[1:-1,2:]
	centre += X[:-2,2:]
	centre += X[:-2,1:-1]

	centre /= (8+centre_weight)


	return(centre)

def RMSE(series1,series2):
	return np.sqrt(np.mean((series1-series2)**2))

def corr_ts(ts_1,ts_2):
	'''
	R = sum( ts_1 x ts_2 ) / np.sqrt( (var(ts_1) x var(ts_2))
	'''

	# detrend both ts 
	ts_1-=(ts_1.mean())
	ts_2-=(ts_2.mean())

	# calculate the numerator of the calc
	numer = np.sum( (ts_1*ts_2))

	# make the denominator terms
	var_ts_1 = np.sum((ts_1**2))
	var_ts_2 = np.sum((ts_2**2))

	# calculate the denominator of the calc
	denom = np.sqrt(var_ts_1*var_ts_2)

	# final 
	R = numer/denom

	return(R)

def p_vals_per_coef(pred, true, coefs, X): 
 sse = sum_squared_error(pred,true)/ float(X.shape[0] - X.shape[1]) 
 standard_error = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))]) 
 t_stats = coefs / standard_error 
 p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), y.shape[0] - X.shape[1])) 
 return p_vals 


def find_nearest(array, value):
    ''' 
    takes array and desired value 
    returns (closest value, index of closest value)
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)


def prob_forecast(ts_training, y_hat, y_hat_training, model_predictors, model_predictors_training):
	'''
	Gives the probability that the storm counts of a season will be within each of the forecast categories: above average, average, below average. Based on a prediction interval with a normal distribution centred on y_hat with SD sigma_hat

	inputs: forecasted storms (y_hat)
			observed storms (ts)
			model_predictors (to calculate the standard deviation of the prediction)

	'''

	from scipy.stats import norm
	import numpy as np 

	y_bar = np.mean(ts_training)
	sigma_bar = np.std(ts_training)
	bot_quart, top_quart = np.percentile(ts_training, [25,75])

	ones_data = np.ones(shape=(len(ts_training)))
	ones = pd.DataFrame(ones_data, columns=['x_o'], index = model_predictors_training.index)
	model_predictors_training_1 = pd.merge(ones,model_predictors_training, left_index=True,right_index=True)
	one = [1]
	one.extend(model_predictors)
	model_predictors_1 = np.array(one)

	MSE = np.mean((y_hat_training.iloc[:,-1]-ts_training)**2)
	inv_xtx = np.linalg.inv(np.matmul(model_predictors_training_1.T,model_predictors_training_1))
	sigma_hat = (np.sqrt(MSE*(1 + np.matmul(np.matmul(model_predictors_1,inv_xtx),model_predictors_1.T))))

	vals = []
	#for th in [y_bar+sigma_bar, y_bar-sigma_bar]:
	for th in [top_quart, bot_quart]:
		vals.append(norm.cdf(th, loc = y_hat, scale = sigma_hat))

	p_above = 1 - vals[0]
	p_below = vals[1]
	p_avg = vals[0]-vals[1]

	return [p_above, p_avg, p_below]

