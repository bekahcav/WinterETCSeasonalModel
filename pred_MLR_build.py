'''
2021 
	Rebekah Cavanagh
	- develop the set of MLRs to forecast storm activity
		- select predictors using stepwise regression and cross-validation
		- fit models and output parameters (best predictors, coefs, etc.)
'''
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sys import path
path.append('/data/home/rebekahc/bek/programs/rc/MSc/tracks_evaluation/')
import eval_functions as ev
path.append('/data/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/')
import pred_functions as pr
from textwrap import wrap

############# load the subseries #############
radius = '750w1000'
subseries = np.load('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_predictands/series_hfx_r'+radius+'_state.npz', allow_pickle=True)['subseries'].item()
keys_series = list(subseries.keys())
years = np.arange(1979,2019)

# use a forecast horizon of 1 month (Sept --> same winter)
# first get the locations of the max corrs over the first 25 years
variables = ['T500','T2M','MSLP','500Z','U250','V250', 'WND250', 'TPWV']
lag = 0

var = 'TPWV'; suff= 'gradient_'; key = 'total'
lat_orig = np.load('/data/home/rebekahc/bek/programs/rc/MSc/tracks_analysis/clim_SOMs/ERA5_dictionaries/'+var+'/clim_grid_'+var+'_r100.npz', allow_pickle=True)['lat']
lon_orig = np.load('/data/home/rebekahc/bek/programs/rc/MSc/tracks_analysis/clim_SOMs/ERA5_dictionaries/'+var+'/clim_grid_'+var+'_r100.npz', allow_pickle=True)['lon']

# choose lat lon range to get predictors from 
lon1,lon2,lat1,lat2 = [257,338,25,63]
clon1 = pr.find_nearest(lon_orig, lon1)[1]
clon2 = pr.find_nearest(lon_orig, lon2)[1]
clat1 = pr.find_nearest(lat_orig, lat2)[1]
clat2 = pr.find_nearest(lat_orig, lat1)[1]

lat = lat_orig[clat1:(clat2+1)]
lon = lon_orig[clon1:(clon2+1)]

len_training = 30 
len_validation = len(years)-len_training 
n_training_sections = 4 
t_cuts = []

for i in range(n_training_sections):
	t_cuts.append((i*len_validation,(i+1)*len_validation))

for key in keys_series:
	print('---------')
	print(key)
	print('---------')

	ts = subseries[key].copy()

	# keep adding more predictor locations until RMSE decrease is small
	top_Y_pred = pd.DataFrame(index=years); top_RMSEs = [100,99,98,97,96]; top_corr = []; model_predictors = pd.DataFrame(index=years); validation_predictors = pd.DataFrame(index=years); regr_best = linear_model.LinearRegression(); counter = 0; no_more_sig = []; override='continue'
	while (model_predictors.shape[1] == counter) and len(no_more_sig)<(len(variables)*2):
		counter+=1
		# determine the best single predictor
		RMSE = 100
		for var in variables:
			for suff in ['gradient_','']:
				try:
					monthly_means = np.load('/data/home/rebekahc/bek/programs/rc/MSc/tracks_analysis/clim_SOMs/ERA5_dictionaries/'+var+'/'+suff+'clim_grid_'+var+'_r100_ASO.npz', allow_pickle=True)[suff+'monthly_means_'+var] #tikoraluk
				except:
					monthly_means = np.load('/data/home/rebekahc/bek/programs/rc/MSc/tracks_analysis/clim_SOMs/ERA5_dictionaries/'+var+'/'+suff+'clim_grid_'+var+'_r100_ASO.npz', allow_pickle=True)[suff+'monthly_means_'] 
				sept_means = monthly_means[clat1:clat2,clon1:clon2,:,1] 
				# remove the linear relationship between the predictors and the candidate predictor
				if len(model_predictors.columns) != 0:
					# fit the predictors and the candidate predictor
					X = model_predictors.copy() #chosen preds

					fit_lines = np.zeros(sept_means.shape)
					for i in range(len(lat)):
						for j in range(len(lon)):
							Y = sept_means[i,j,:] #candidate predictors

							regr_pred = linear_model.LinearRegression()
							regr_pred.fit(X, Y)

							# determine portion of sept means linearly related to existing predictors (correlated part)
							fitted_sept_means = regr_pred.predict(X)
							fit_lines[i,j] = fitted_sept_means

					# subtract the colinearity (correlated part)
					fitted_sept_means = sept_means - fit_lines

				else:
					fitted_sept_means = sept_means

				# calculate corr coefficient of field  
				R = pr.corrcoef_map(fitted_sept_means, ts) 

				# use the locations with statistically significant correlations
				sig_thresh = 1.96/(np.sqrt(len(ts)))
				sig_lats = np.where(np.abs(R)>sig_thresh)[0]
				sig_lons = np.where(np.abs(R)>sig_thresh)[1]
				order = np.argsort(np.abs(R[sig_lats,sig_lons]))[::-1]

				if len(sig_lats) == 0:
					print('no more significant variables in ',suff,var)
					if (suff+var) not in no_more_sig:
						no_more_sig.append(suff+var)

				for i in range(len(sig_lats)):
					# only evaluate if it hasn't already been selected as a predictor 
					name = suff+var+' '+str((lat[sig_lats[i]],lon[sig_lons[i]]))
					if name not in model_predictors.columns:
						temp_SqE = np.array([])
						temp_corr = 0
						Y_pred_full = pd.DataFrame()
						for n in range(n_training_sections):
							t_cut1, t_cut2 = t_cuts[n]

							X = model_predictors[:t_cut1].append(model_predictors[t_cut2:])
							X[(lat[sig_lats[i]],lon[sig_lons[i]])] = np.append(sept_means[sig_lats[i],sig_lons[i],:t_cut1],sept_means[sig_lats[i],sig_lons[i],t_cut2:]).T

							Y = np.append(ts[:t_cut1],ts[t_cut2:])

							regr = linear_model.LinearRegression()
							regr.fit(X, Y)

							#print('Intercept: \n', regr.intercept_)
							#print('Coefficients: \n', regr.coef_)

							# prediction over validation period
							val_sept_means = validation_predictors.copy()[t_cut1:t_cut2]
							val_sept_means[(lat[sig_lats[i]],lon[sig_lons[i]])] = sept_means[sig_lats[i],sig_lons[i],t_cut1:t_cut2].T

							#print ('Predicted Storms: \n', regr.predict(val_sept_means))

							# evaluate the prediction
							Y_pred = regr.predict(val_sept_means)
							Y_pred_full = Y_pred_full.append(pd.DataFrame(Y_pred,index=years[t_cut1:t_cut2],columns=['storms']))
							Y_actual = ts[t_cut1:t_cut2]

							temp_SqE = np.append(temp_SqE,((Y_pred-Y_actual)**2))
							temp_corr += pr.corr_ts(Y_pred, Y_actual)
							ts = subseries[key].copy()

						# calculate the RMSE of the predictor over all the training cycles
						temp_RMSE = np.sqrt(np.mean(temp_SqE))
						temp_corr/=n_training_sections

						if (temp_RMSE < RMSE):
							#print('new best RMSE:',temp_RMSE, RMSE)
							RMSE = temp_RMSE
							best_corr = temp_corr
							best_name = suff+var+' '+str((lat[sig_lats[i]],lon[sig_lons[i]]))
							best_means = sept_means[sig_lats[i],sig_lons[i]]
							best_pred = Y_pred_full.sort_index()
							X = model_predictors.copy()
							X[(lat[sig_lats[i]],lon[sig_lons[i]])] = sept_means[sig_lats[i],sig_lons[i],:].T
							regr_best.fit(X,ts)

					else:
						print('predictor already added')

				# if it makes it this far and hasn't found another predictor the RMSE will still be 100. In such a case we have determined there are no more possible predictors to add so we should add this suff+var to the list of no_more_sig
				if RMSE == 100:
					print('all significant variables already included in ',suff,var)
					print(name)
					if (suff+var) not in no_more_sig:
						no_more_sig.append(suff+var)					
		# if it makes it this far and hasn't found a predictor that makes the model any better, the RMSE will be greater than the last 

		print('next candidate: ',best_name, 'RMSE: ',RMSE)

		##### don't accept to the model if it makes the RMSE worse 
		if (RMSE > top_RMSEs[-1]):
			print('*** addition of predictor increases RMSE ***')
		## if we found a new predictor
		#if RMSE !=100 :
		if (RMSE < top_RMSEs[-1]):
			# add the best predictor location to the model 
			model_predictors[best_name] = best_means.T
			validation_predictors[best_name] = best_means.T
			#validation_predictors[best_name] = np.append(best_means[:t_cut1],best_means[t_cut2:]).T
			co = str(counter)+' predictors'
			top_Y_pred[co] = best_pred

			top_RMSEs.append(RMSE)
			top_corr.append(best_corr)

		print('-----', model_predictors.shape[1], 'predictors selected')
		print(model_predictors.columns)

	print('model_predictors shape',model_predictors.shape[1],'top_RMSEs[-1]',top_RMSEs[-1],'no more sig in', no_more_sig)

	print(top_RMSEs)
	np.savez('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_MLR/parameters_'+key+'_Sept_multitrained_sub2.npz',top_RMSEs=top_RMSEs[5:],top_Y_pred=top_Y_pred, top_corr=top_corr, intercept=regr_best.intercept_, coefficients=regr_best.coef_)
	top_Y_pred.to_pickle('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_MLR/predicted_values'+key+'_Sept_multitrained_sub2.pkl')
	model_predictors.to_pickle('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_MLR/model_predictors'+key+'_Sept_multitrained_sub2.pkl')
	validation_predictors.to_pickle('/home/rebekahc/bek/programs/rc/MSc/tracks_prediction/pred_MLR/validation_predictors'+key+'_Sept_multitrained_sub2.pkl')


