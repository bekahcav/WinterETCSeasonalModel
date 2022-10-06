'''
2021 Feb 18

	Rebekah Cavanagh

	get obs from ECCC archive for Halifax and process them

for year in `seq 1979 2011`;do for month in `seq 1 12`;do wget --content-disposition "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=6358&Year=${year}&Month=${month}&Day=14&timeframe=1&submit= Download+Data" ;done;done

for year in `seq 2013 2020`;do for month in `seq 1 12`;do wget --content-disposition "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=50620&Year=${year}&Month=${month}&Day=14&timeframe=1&submit= Download+Data" ;done;done
'''
import pandas as pd 
import matplotlib.pyplot as plt 

# cd to data location

# start with this to get correct cols etc. but we will drop this entry at the end
hfx_obs = pd.read_csv('./en_climate_hourly_NS_8202250_10-1979_P1H.csv', index_col = ['Date/Time (LST)'])

# add in the first station's obs
yrs = range(1979, 2012)
mns = ['11','12','01','02','03']
for yr in yrs:
	for mn in mns[:2]:
		df_next = pd.read_csv('./en_climate_hourly_NS_8202250_'+mn+'-'+str(yr)+'_P1H.csv', index_col = ['Date/Time (LST)'])
		hfx_obs = hfx_obs.append(df_next)
	for mn in mns[2:]:
		df_next = pd.read_csv('./en_climate_hourly_NS_8202250_'+mn+'-'+str(yr+1)+'_P1H.csv', index_col = ['Date/Time (LST)'])
		hfx_obs = hfx_obs.append(df_next)


# then # add in the second station's obs
yrs = range(2012, 2019)
mns = ['11','12','01','02','03']
for yr in yrs:
	for mn in mns[:2]:
		df_next = pd.read_csv('./en_climate_hourly_NS_8202251_'+mn+'-'+str(yr)+'_P1H.csv', index_col = ['Date/Time (LST)'])
		hfx_obs = hfx_obs.append(df_next)
	for mn in mns[2:]:
		df_next = pd.read_csv('./en_climate_hourly_NS_8202251_'+mn+'-'+str(yr+1)+'_P1H.csv', index_col = ['Date/Time (LST)'])
		hfx_obs = hfx_obs.append(df_next)

hfx_obs.index = pd.to_datetime(hfx_obs.index).tz_localize('Etc/GMT-4')

hfx_obs = hfx_obs[744:]

# plot monthly mean
means = []
for mn in mns:
	means.append(hfx_obs.loc[hfx_obs['Month'] == int(mn)]['Temp (°C)'].mean())

plt.plot(['nov','dec','jan','feb','mar'], means, c='lightskyblue')  
plt.scatter(['nov','dec','jan','feb','mar'], means, c='lightskyblue') 

# Now test annual winter temp by season (consecutive Nov-Mar months)
szn_means = []
for yr in range(1979,2019):
	nov = (hfx_obs['Month']==11)&(hfx_obs['Year']==yr)
	dec = (hfx_obs['Month']==12)&(hfx_obs['Year']==yr)
	jan = (hfx_obs['Month']==1)&(hfx_obs['Year']==yr+1)
	feb = (hfx_obs['Month']==2)&(hfx_obs['Year']==yr+1)
	mar = (hfx_obs['Month']==3)&(hfx_obs['Year']==yr+1)

	avg = hfx_obs[nov|dec|jan|feb|mar]['Temp (°C)'].mean()
	szn_means.append(avg)


hfx_obs.to_pickle('./hfx_obs_compiled_1979-2019.pkl')