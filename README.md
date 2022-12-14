# Winter Seasonal Extratropical Cyclone Forecast Model
This respository contains six main scripts and three supporting scripts that together produce a probabilistic seasonal forecast of extratropical cyclone (ETCs) activity and characteristics at a given location (I used Halifax). It has two main sections, the first being the development of an ETC storm track dataset and the second being the development of a multiple linear regression (MLR) model and a probabilistic framework that gives forecasts based on the MLR outputs.   

The Tracking section executes the automated detection and tracking of low-pressure storms from a series of mean sea level pressure (MSLP) maps from ERA5 Reanalysis dataset. Developed as an adaptation of [stormTracking by Dr. Eric Oliver](https://github.com/ecjoliver/stormTracking) with added criteria for storm detection and slightly modified method of defining the storm centre.

The Prediction section executes the development of a multiple linear regression (MLR) model. First the predictand is defined and then the predictors are selected using stepwise regression and cross-validation. Next, the models are trained and then the MLR outputs (prediction and prediction distribution) are used to give a probabilistic forecast of the seasonal ETC activity.

## Process  

### Tracking
1. [storm_detection.py](https://github.com/bekahcav/WinterETCSeasonalModel/blob/main/storm_detection.py)
    - detects possible storm centres from ERA5 MSLP fields
        - input: [ERA5 MSLP data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)
        - output: locations of detected storm centres
2. [storm_tracking.py](https://github.com/bekahcav/WinterETCSeasonalModel/blob/main/storm_tracking.py)
    - connects possible storm centres through time to build 
        - input: locations of detected storm centres
        - output: list of dictionaries of storm tracks


### Prediction
3. [pred_observations.py](https://github.com/bekahcav/WinterETCSeasonalModel/blob/main/pred_observations.py)
    - obtain and process the ECCC station data
        - input: csv of ECCC obs from Halifax 
        - output: Pandas database of ECCC obs from Halifax
4. [pred_hfxstorms.py](https://github.com/bekahcav/WinterETCSeasonalModel/blob/main/pred_hfxstorms.py)
    - select the storms that affect Halifax to build a prediction timeseries
    - split the storms into precip (rain/snow/mixed), no precip, high wind, and bomb storms
    - save dictionaries with the subseries of storms
        - input: list of dictionaries of storm tracks, database of ECCC observations
        - output: 8 type-specific predictand time series
5. [pred_MLR_build.py](https://github.com/bekahcav/WinterETCSeasonalModel/blob/main/pred_MLR_build.py)
    - predictors selected using stepwise regression and cross-validation
    - MLR is trained 
        - input: [ERA5 possible predictor fields](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) 
        - output: final predictors, regression coefficients of fitted MLRs
6. [pred_forecast.py](https://github.com/bekahcav/WinterETCSeasonalModel/blob/main/pred_forecast.py)
    - predict the number of storms using the MLR
    - calculate the probability of average, below average, and above average storm activity 
    - produce the worded probabilist forecast
        - input: ERA5 predictor fields for year being forecast (sept avg), MLR equations
        - output: probabilistic forecast of winter ETC activity, MLR-predicted number of storms 
        
        
## 
This work forms part of my [MSc thesis](https://dalspace.library.dal.ca/handle/10222/81485) completed at Dalhousie University under the supervision of [Dr. Eric Oliver](https://github.com/ecjoliver). If you have any questions, you can reach me by email at r.cavanagh@dal.ca.
