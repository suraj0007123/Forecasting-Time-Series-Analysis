import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from statsmodels.tsa.seasonal import seasonal_decompose # tsa = Time series analysis
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing


# load the data
data = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\forecasting time series\Datasets_Forecasting\solarpower_cumuldaybyday2.csv")
data.columns  
data.head()

data.cum_power.plot() # time series plot 
# upward trend.

# Splitting the data into Train and Test data
# take last 30 days  as test
2558-30
Train = data.head(2528)
Test = data.tail(30)  

# Mean absolute percentage error = MAPE
# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org): # org = original
    temp = np.abs((pred-org)/org)*100  # absolut (error diveded by original values) *100
    return np.mean(temp)


# Moving Average for the time series
mv_pred = data["cum_power"].rolling(30).mean() # takes widow width= 30 values to calculate MA.
# A rolling mean is simply the mean of a certain number of previous periods in a time series.
mv_pred.tail(30)  # last 30 MA values
MAPE(mv_pred.tail(30), Test.cum_power)*100 # % error



# Plot with Moving Averages
data.cum_power.plot(label = "org")
for i in range(2, 31, 2): # taking different window width  as 2,4,6,8. 
    data["cum_power"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
# WW = all give straight line, eliminating seasonality
# Run all in one go. above 4 code lines.
# taking 30 as Moving average period.

# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(data.cum_power, model = "additive", period = 30)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(data.cum_power, model = "multiplicative", period = 30)
decompose_ts_mul.plot()



# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(data.cum_power, lags = 30)
tsa_plots.plot_pacf(data.cum_power, lags= 30) 
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series




# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["cum_power"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1]) # taking indexing values.
MAPE(pred_ses, Test.cum_power)*100


# Holt method 
hw_model = Holt(Train["cum_power"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.cum_power)*100 


# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 30).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.cum_power)*100
# least error


# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["cum_power"], seasonal = "mul", trend = "add", seasonal_periods = 30).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.cum_power)*100


### Holts winter Exponential smoothing model gives less error. Hence this is the best model.