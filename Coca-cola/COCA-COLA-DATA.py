import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

# load the data
data = pd.read_excel(r"E:\DESKTOPFILES\suraj\assigments\forecasting time series\Datasets_Forecasting\CocaCola_Sales_Rawdata.xlsx")
data.columns  
data.head()

# Quarter = one quater of year out of 4 ( it has three months),
# sales = sales of cocacola items.
data.info()

#### Pre-Processing
# take sequential data 
data['t']=np.arange(1, 43)

# create all variables that we need to put in all foracasting equations.
data['t_sqr'] = data['t']*data['t'] # Need this to form quaderatic equation.
data['log_Sales'] = np.log(data.Sales) # Need this to form exponential and multiplicative seasonality.
data.columns 


# will take only quarter names in seperate column
p = data.Quarter[0]
p[:2]
data['quarter'] = 0
for i in range(0, 42):
    p = data.Quarter[i]
    data['quarter'][i] = p[:2]

# Creating dummies for quarter column

quarter_dummies = pd.DataFrame(pd.get_dummies(data['quarter']))
# add this to main data.  this will be usefule to get seasonality.
data = pd.concat([data, quarter_dummies], axis=1)

##### upto this we created the necessary data.

# Data preprocessing and partition

# visualization - time plot
data.Sales.plot()
# from plot: level = increasing,
#            trend = upward.
#            seasonality is subjective cannot be defined from plot

# lets do quantification = try to fit data in different equations. 
# then we check the errors. LESSER THE ERROR, BETTER THE MODEL.

# data partition:
# 1 season = 4 quarters
# remove last two quarters
data1 = data.drop(index=[40, 41], axis=0)

train = data1.head(36) # we need sequencial data hence we take head 36 value for training
test = data1.tail(4) # last one season of 4 quarters for test. 


# Try multiple models

######### LINEAR #########3
import statsmodels.formula.api as smf
linear_model = smf.ols('Sales ~ t', data=train).fit()
pred_linear = pd.Series(linear_model.predict(test.t))

rmse_linear = np.sqrt(np.mean((test.Sales-pred_linear)**2))
rmse_linear 

######### Exponential  #########
exp_model = smf.ols('log_Sales ~ t', data=train).fit()
pred_exp = pd.Series(exp_model.predict(test.t))

rmse_exp = np.sqrt(np.mean((test.Sales-np.exp(pred_exp))**2))
rmse_exp 


######### Quadratic   #########
quad_model = smf.ols('Sales ~ t+t_sqr', data=train).fit()
pred_quad = pd.Series(quad_model.predict(test[['t','t_sqr']]))

rmse_quad = np.sqrt(np.mean((test.Sales-(pred_quad))**2))
rmse_quad 

######### Additive Seasonality   #########
add_sea_model = smf.ols('Sales ~ Q1+Q2+Q3+Q4', data=train).fit()
pred_add_sea = pd.Series(add_sea_model.predict(test[['Q1','Q2','Q3','Q4']]))

rmse_add_sea = np.sqrt(np.mean((test.Sales-(pred_add_sea))**2))
rmse_add_sea 

######### Multiplicative  Seasonality   #########
mul_sea_model = smf.ols('log_Sales ~ Q1+Q2+Q3+Q4', data=train).fit()
pred_mul_sea = pd.Series(mul_sea_model.predict(test[['Q1','Q2','Q3','Q4']]))

rmse_mul_sea = np.sqrt(np.mean((test.Sales-np.exp(pred_mul_sea))**2))
rmse_mul_sea 


######### Additive Seasonality Quadratic Trend    #########
add_sea_quad_model = smf.ols('Sales ~ t+t_sqr+Q1+Q2+Q3+Q4', data=train).fit()
pred_add_quad_sea = pd.Series(add_sea_quad_model.predict(test[['Q1','Q2','Q3','Q4','t','t_sqr']]))

rmse_add_quad_sea = np.sqrt(np.mean((test.Sales-(pred_add_quad_sea))**2))
rmse_add_quad_sea 


######### Multiplicative Seasonality Linear Trend    #########
mul_sea_lin_model = smf.ols('Sales ~ t+Q1+Q2+Q3+Q4', data=train).fit()
pred_mul_lin_sea = pd.Series(mul_sea_lin_model.predict(test))

rmse_mul_lin_sea = np.sqrt(np.mean((test.Sales-(pred_mul_lin_sea))**2))
rmse_mul_lin_sea 

################## Prediction ##################################3


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),
        "RMSE_Values":pd.Series([rmse_linear, rmse_exp, pred_quad , rmse_add_sea, rmse_mul_sea, rmse_add_quad_sea, rmse_mul_lin_sea])}
table_rmse = pd.DataFrame(data)
table_rmse

# Prediction for next two years 

# Here Additive Seasonality Quanderatic trend give least error.
add_sea_quad_model = smf.ols('Sales ~ t+t_sqr+Q1+Q2+Q3+Q4', data=data1).fit()
#declare new dataframe
two_year_data = {'t':[41,42,43,44,45,46,47,48], 'quarter': ['Q1','Q2','Q3','Q4','Q1','Q2','Q3','Q4'] }
test_data = pd.DataFrame(two_year_data)
test_data['t_sqr'] = test_data['t']*test_data['t']

# Create the data to predict.
quarter_dummies = pd.DataFrame(pd.get_dummies(test_data['quarter']))
# add this to main data.  this will be usefule to get seasonality.
data2 = pd.concat([test_data, quarter_dummies], axis=1)

# Prediction for two years.
pred_add_quad_sea = pd.Series(add_sea_quad_model.predict(data2[['Q1','Q2','Q3','Q4','t','t_sqr']]))

print(pred_add_quad_sea) # prediction for two year.

