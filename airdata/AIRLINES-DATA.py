import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

# load the data
data = pd.read_excel(r'E:\DESKTOPFILES\suraj\assigments\forecasting time series\Datasets_Forecasting\Airlines Data.xlsx')
data.columns  
data.head()

# Month = date (a measure of time),
# Passenger = No. of people flying in airlines.
data.info()

#### Pre-Processing

# since month are in datetime format, hence,
# creating new column for time with element from 1 to 159 in it.
# these numbers are in sequencial format like months.
# here, time is for trend and months for seasonality
data['t']=np.arange(1, 97)

# create all variables that we need to put in all foracasting equations.
data['t_sqr'] = data['t']*data['t'] # Need this to form quaderatic equation.
data['log_passengers'] = np.log(data.Passengers) # Need this to form exponential and multiplicative seasonality.
data.columns 

# Assuming date format as yyyy-mm-dd and data is collected on first day of each month in years 1995-02.
# Getting month names in new columns 
months = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
months_8 = months*8 # there are 8 year in the seasons

data['months'] = months_8

# Creating dummies for months column
# data = pd.get_dummies(data, columns=['months']) # Months column no longer exists 
# taking seperate month dummies dataframe
month_dummies = pd.DataFrame(pd.get_dummies(data['months']))
# add this to main data.  this will be usefule to get seasonality.
data = pd.concat([data, month_dummies], axis=1)

##### upto this we created the necessary data.

# Data preprocessing and partition

# visualization - time plot
data.Passengers.plot()
# from plot: level = increasing,
#            trend = upward.
#            seasonality is subjective cannot be defined from plot

# lets do quantification = try to fit data in different equations. 
# then we check the errors. LESSER THE ERROR, BETTER THE MODEL.

# data partition:
# data in season of 12 months Hence we should predict at least 1 season. ie. 12 Months
96-12
train = data.head(84) # we need sequencial data hence we take head 84 value for training
test = data.tail(12) # last one season for test. 


# Try multiple models

######### LINEAR #########3
import statsmodels.formula.api as smf
linear_model = smf.ols('Passengers ~ t', data=train).fit()
pred_linear = pd.Series(linear_model.predict(test.t))

rmse_linear = np.sqrt(np.mean((test.Passengers-pred_linear)**2))
rmse_linear 

######### Exponential  #########
exp_model = smf.ols('log_passengers ~ t', data=train).fit()
pred_exp = pd.Series(exp_model.predict(test.t))

rmse_exp = np.sqrt(np.mean((test.Passengers-np.exp(pred_exp))**2))
rmse_exp 


######### Quadratic   #########
quad_model = smf.ols('Passengers ~ t+t_sqr', data=train).fit()
pred_quad = pd.Series(quad_model.predict(test[['t','t_sqr']]))

rmse_quad = np.sqrt(np.mean((test.Passengers-(pred_quad))**2))
rmse_quad 

######### Additive Seasonality   #########
add_sea_model = smf.ols('Passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec', data=train).fit()
pred_add_sea = pd.Series(add_sea_model.predict(test[['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))

rmse_add_sea = np.sqrt(np.mean((test.Passengers-(pred_add_sea))**2))
rmse_add_sea 

######### Multiplicative  Seasonality   #########
mul_sea_model = smf.ols('log_passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec', data=train).fit()
pred_mul_sea = pd.Series(mul_sea_model.predict(test[['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))

rmse_mul_sea = np.sqrt(np.mean((test.Passengers-np.exp(pred_mul_sea))**2))
rmse_mul_sea 


######### Additive Seasonality Quadratic Trend    #########
add_sea_quad_model = smf.ols('Passengers ~ t+t_sqr+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec', data=train).fit()
pred_add_quad_sea = pd.Series(add_sea_quad_model.predict(test[['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t','t_sqr']]))

rmse_add_quad_sea = np.sqrt(np.mean((test.Passengers-(pred_add_quad_sea))**2))
rmse_add_quad_sea 


######### Multiplicative Seasonality Linear Trend    #########
mul_sea_lin_model = smf.ols('Passengers ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec', data=train).fit()
pred_mul_lin_sea = pd.Series(mul_sea_lin_model.predict(test))

rmse_mul_lin_sea = np.sqrt(np.mean((test.Passengers-(pred_mul_lin_sea))**2))
rmse_mul_lin_sea 

################## Testing ##################################


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear, rmse_exp, pred_quad , rmse_add_sea, rmse_mul_sea, rmse_add_quad_sea, rmse_mul_lin_sea])}
table_rmse = pd.DataFrame(data)
table_rmse

# Here Additive Seasonality Quanderatic trend give least error.