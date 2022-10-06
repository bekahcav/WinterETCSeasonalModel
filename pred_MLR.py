'''
output:
	predicted number of each type of storms for the season with likelihood in each prediction category.
	df - columns = storm categories
	rows = y_hat, prob above avg, prob avg, prob below avg
'''

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from sys import path
path.append('/data/home/rebekahc/bek/programs/rc/MSc/tracks_evaluation/')
import eval_functions as ev
path.append('/data/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/')
import pred_functions as pr
from textwrap import wrap
from sklearn import linear_model
import statsmodels.api as sm

def norm_pdf(X, mean, SD):
	fx = (1/(SD*(np.sqrt(2*np.pi))))*np.exp(-((X-mean)/SD)**2/2)
	return()

radius = '750w1000'
subseries = np.load('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_predictands/series_hfx_r'+radius+'_state_pred.npz', allow_pickle=True)['subseries'].item()
subseries_training = np.load('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_predictands/series_hfx_r'+radius+'_state.npz', allow_pickle=True)['subseries'].item()
keys_series = list(subseries.keys())
colours = {'total': 'slategrey', 'total_precip': 'olive', 'snow_count': 'lightskyblue', 'rain_count': 'lightcoral', 'mixed_count': 'thistle', 'no_precip_count': 'tan', 'wind_count': 'steelblue', 'bomb_count': 'mediumturquoise'}
lat_orig = np.load('/data/home/rebekahc/bek/programs/rc/MSc/tracks_analysis/clim_SOMs/ERA5_dictionaries/T2M/clim_grid_T2M_r100_unscaled.npz', allow_pickle=True)['lat']
lon_orig = np.load('/data/home/rebekahc/bek/programs/rc/MSc/tracks_analysis/clim_SOMs/ERA5_dictionaries/T2M/clim_grid_T2M_r100_unscaled.npz', allow_pickle=True)['lon']
lon1,lon2,lat1,lat2 = [257,338,25,63]
clon1 = pr.find_nearest(lon_orig, lon1)[1]
clon2 = pr.find_nearest(lon_orig, lon2)[1]
clat1 = pr.find_nearest(lat_orig, lat2)[1]
clat2 = pr.find_nearest(lat_orig, lat1)[1]

lat = lat_orig[clat1:(clat2+1)]
lon = lon_orig[clon1:(clon2+1)]

########################
# 	  final models 	 #
########################


# load the model details
yearStart = 2019; yearEnd = 2022; years = np.arange(yearStart, yearEnd)
for yr in years:
	yr_ind = yr-2019
	forecast = pd.DataFrame(columns = keys_series, index = ['y_obs','y_hat', 'P(above)', 'P(average)', 'P(below)','sigma_hat'])
	for key_i in range(len(keys_series)):
		key = keys_series[key_i]
		ts = subseries[key]
		ts_training = subseries_training[key]
		colour = colours[key]
		model_predictors_training = pd.read_pickle('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_MLR/model_predictors'+key+'_Sept_multitrained_sub2_980665.pkl')
		full_parameters = np.load('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_MLR/parameters_'+key+'_Sept_multitrained_sub2_980665.npz',allow_pickle=True); coefs = full_parameters['coefficients']; intcpt = full_parameters['intercept']
		y_hat_training = pd.read_pickle('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_MLR/predicted_values'+key+'_Sept_multitrained_sub2_980665.pkl')
		model = []
		if yr_ind<len(ts) :
			forecast_key = [(ts[yr_ind])]
		else:
			forecast_key = ['-']

		# for each predictor, load the predictor field and extract the values
		for predictor in model_predictors_training.columns:
			field, coords = predictor.split(' ',1)
			if field[:8] == 'gradient':
				suff, var = field.split('_')
				suff+='_'
			else:
				var = field
				suff=''
			coords = coords[1:-1].split(', ')
			lat_i = float(coords[0]); lon_i = float(coords[1])

			try:
				monthly_means = np.load('/data/home/rebekahc/bek/programs/rc/MSc/tracks_analysis/clim_SOMs/ERA5_dictionaries/'+var+'/'+suff+'clim_grid_'+var+'_r100_ASO_new.npz', allow_pickle=True)[suff+'monthly_means_'+var] #tikoraluk
			except:
				monthly_means = np.load('/data/home/rebekahc/bek/programs/rc/MSc/tracks_analysis/clim_SOMs/ERA5_dictionaries/'+var+'/'+suff+'clim_grid_'+var+'_r100_ASO_new.npz', allow_pickle=True)[suff+'monthly_means_'] 

			sept_means = monthly_means[clat1:clat2,clon1:clon2,yr_ind,1] 

			lat_ind = np.argwhere(lat==[lat_i])[0][0]
			lon_ind = np.argwhere(lon==[lon_i])[0][0]

			model.append(sept_means[lat_ind, lon_ind])

		y_hat = intcpt + (coefs*model).sum()
		forecast_key.append(y_hat)

		# calculate the probabilities of each cat
		probs = pr.prob_forecast(ts_training, y_hat, y_hat_training, model, model_predictors_training)
		forecast_key.extend(probs)
    pr.worded_forecast(ts_training, y_hat, y_hat_training, model, model_predictors_training)

		forecast[key] = forecast_key
		#print(y_hat,model,np.mean(model_predictors_training),sep='\n')

	print(forecast)
	forecast.to_pickle('/data/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_MLR/pred_outputs/forecasts/stormprediction_'+str(yr)+'-'+str(yr+1)+'season_980665')


for yr in years:
	print(' ',yr,sep ='\n')
	forecast = pd.read_pickle('/data/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_MLR/pred_outputs/forecasts/stormprediction_'+str(yr)+'-'+str(yr+1)+'season_980665')
	print(forecast.round(3))

