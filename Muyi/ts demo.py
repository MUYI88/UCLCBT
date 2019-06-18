#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 21:04:17 2019

@author: guisier
"""


import pandas as pd
import numpy as np
from cryptocmd import CmcScraper

#ripple data 5 years
# initialise scraper without time interval
scraper = CmcScraper("XRP","01-06-2014", "31-05-2019")
# get raw data as list of list
headers, data = scraper.get_data()
# get data in a json format
xrp_json_data = scraper.get_data("json")
# export the data as csv file, you can also pass optional `name` parameter
scraper.export("csv", name="xrp_all_time")
# Pandas dataFrame for the same data
df = scraper.get_dataframe()

#bitcoin data 5 years
scraper1 = CmcScraper("BTC","01-06-2014", "31-05-2019")
# get raw data as list of list
headers, data = scraper1.get_data()
# get data in a json format
xrp_json_data = scraper1.get_data("json")
# export the data as csv file, you can also pass optional `name` parameter
scraper1.export("csv", name="btc_all_time")
# Pandas dataFrame for the same data
dfB = scraper1.get_dataframe()

# daily price figure
Date=df['Date']
Close=df['Close']
DateB=dfB['Date']
CloseB=dfB['Close']
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(Date, Close,linewidth = '0.5',color='b',label='Ripple')
plt.legend(loc=2)
ax1.tick_params(axis='y', colors='b')
ax1.set_ylabel('Ripple',color='b')

ax1.set_title('Daily closing price $')
ax2 = ax1.twinx()  # this is the important function
ax2.plot(DateB, CloseB, linewidth = '0.5',color='r',label='Bitcoin')
plt.legend(loc=1)
ax2.tick_params(axis='y', colors='r')
ax2.set_ylabel('Bitcoin',color='r')
plt.show()

# daily log return figure
df = scraper.get_dataframe()
dfdaily=df[::-1]
dfdaily=dfdaily[['Date','Close']]
dfdaily['log_price'] = np.log(dfdaily.Close)
dfdaily['log_ret'] = dfdaily.log_price.diff()
dfdaily=dfdaily.dropna(axis=0,how='any')
log_ret_R=dfdaily['log_ret']

dfB = scraper1.get_dataframe()
dfdailyB=dfB[::-1]
dfdailyB=dfdailyB[['Date','Close']]
dfdailyB['log_price_B'] = np.log(dfdailyB.Close)
dfdailyB['log_ret_B'] = dfdailyB.log_price_B.diff()
dfdailyB=dfdailyB.dropna(axis=0,how='any')
log_ret_B=dfdailyB['log_ret_B']


# weekly log return figure
df = scraper.get_dataframe()
dfweek=df[::-1]
dfweek=dfweek[::7]
dfweek=dfweek[['Date','Close']]
dfweek['log_price_weekR'] = np.log(dfweek.Close)
dfweek['log_ret_weekR'] = dfweek.log_price_weekR.diff()
dfweek=dfweek.dropna(axis=0,how='any')
log_ret_weekR=dfweek['log_ret_weekR']

dfB = scraper1.get_dataframe()
dfweekB=dfB[::-1]
dfweekB=dfweekB[::7]
dfweekB=dfweekB[['Date','Close']]
dfweekB['log_price_weekB'] = np.log(dfweekB.Close)
dfweekB['log_ret_weekB'] = dfweekB.log_price_weekB.diff()
dfweekB=dfweekB.dropna(axis=0,how='any')
log_ret_weekB=dfweekB['log_ret_weekB']

Date=dfdaily['Date']
DateB=dfdailyB['Date']
DateWR=dfweek['Date']
DateWB=dfweekB['Date']

plt.subplot(211)
plt.title('Daily log-return')
plt.plot(Date,log_ret_R,linewidth = '0.3',color='b',label='Ripple',alpha=0.5)
plt.plot(DateB,log_ret_B,linewidth = '0.4',color='r',label='Bitcoin',alpha=0.7)
plt.legend()
plt.subplot(212)
plt.title('Weekly log-return')
plt.plot(DateWR,log_ret_weekR,linewidth = '0.5',color='b',label='Ripple')
plt.plot(DateWB,log_ret_weekB,linewidth = '0.5',color='r',label='Bitcoin')
plt.legend()
plt.tight_layout()
plt.show()

# statistical Properties for 5 years daily log-returns
np.mean(log_ret_R)
np.std(log_ret_R)
np.min(log_ret_R)
np.max(log_ret_R)
log_ret_R.skew(axis=0)
log_ret_R.kurt(axis=0)

np.mean(log_ret_B)
np.std(log_ret_B)
np.min(log_ret_B)
np.max(log_ret_B)
log_ret_B.skew(axis=0)
log_ret_B.kurt(axis=0)

# use first 2 years to fit model
#ripple data 2 years
log_ret=log_ret_R[0:730]

#check the stationary
from statsmodels.tsa.stattools import adfuller
result = adfuller(log_ret, autolag='AIC')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
# We can see that our statistic value of -11.756 is less than the value of -3.439 
#at 1% or 5% or 10%. This suggests that we can reject the null hypothesis with 
#a significance level of less than 1% or 5% or 10% 
#Rejecting the null hypothesis means that the process has no unit root, 
#and in turn that the time series is stationary or does not have time-dependent structure.
        
# 分解decomposing, make sure it's no trend and seaonality
from statsmodels.tsa.seasonal import seasonal_decompose
res = seasonal_decompose(log_ret, model='additive',freq = 1)
res.plot()
plt.show()

#白噪声检验,p value 小于0.5 not white noise
from statsmodels.stats.diagnostic import acorr_ljungbox
#返回统计量和p值
noiseRes = acorr_ljungbox(log_ret, lags=1)
print('stat                  | p-value')
for x in noiseRes:
    print(x,'|', end=" ")


from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
plt.figure(figsize=(10,10))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(log_ret, lags=60, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(log_ret, lags=60, ax=ax)
plt.tight_layout()
plt.show()

# statsmodels.tsa.stattools.arma_order_select_ic(y, max_ar=4, max_ma=2, ic='bic', trend='c', model_kw={}, fit_kw={})
from statsmodels.tsa.stattools import arma_order_select_ic
res = arma_order_select_ic(log_ret,ic=['aic','bic'],trend='nc')
# res.aic_min_order
res.bic_min_order
res.aic_min_order

log_ret=log_ret_R[0:730]
# use middle 2 years to validate model，choose the best
#ripple data middle 2 years
log_ret_34=log_ret_R[730:1460]


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from math import sqrt
from sklearn.metrics import mean_squared_error
train = list(log_ret)
test = list(log_ret_34)
predictions = []
history = [x for x in train]
for i in range(len(test)):
    # make prediction
    model = ARMA(history, order=(0,1))
    model_fit = model.fit(disp=-1)
    pred = model_fit.forecast()[0]
    predictions.append(pred)
    # extract real value
    obs = test[i]
    # from this moment real value becomes history which will be used as the prediction for next iteration
    history.append(obs)
# Prediction Accuracy
mse = mean_squared_error(test, predictions)
mse_r = sqrt(mse)
print('SQUARE ROOT OF MSE: %.5f' % mse_r)


#check date, not use in code
log_ret_RT=dfdaily[['Date','log_ret']]
log_ret_RT.set_index("Date", inplace=True)
log_ret_RT

#predict, week by week
len(log_ret_R)
train = list(log_ret_R[:1460])
train
test = log_ret_R[len(train):len(train)+7]
test
len(test)

rmse = []
history = [x for x in train]
count = 1
predictions_result = []
while count < 53:
    test = list(log_ret_R[len(history):len(history)+7])
    predictions = []
    for i in range(len(test)):
    # make prediction
        model = ARMA(history, order=(4,0))
        model_fit = model.fit(disp=-1)
        pred = model_fit.forecast()[0]
        predictions.append(pred)
        predictions_result.append(pred)
    # extract real value
        obs = test[i]
    # from this moment real value becomes history which will be used as the prediction for next iteration
        history.append(obs)
# Prediction Accuracy
    rmse.append(sqrt(mean_squared_error(test, predictions)))
    count += 1
    
rmse
predictions_result

plt.plot(predictions_result)
plt.plot(log_ret_R[1460:1824])

#predict, day by day
len(log_ret_R)
train = list(log_ret_R[:1460])
train
test = log_ret_R[len(train):len(train)+1]
test
len(test)

rmse = []
history = [x for x in train]
count = 1
predictions_result = []
while count < 366:
    test = list(log_ret_R[len(history):len(history)+1])
    predictions = []
    for i in range(len(test)):
    # make prediction
        model = ARIMA(history, order=(4,0,0))
        model_fit = model.fit(disp=-1)
        pred = model_fit.forecast()[0]
        predictions.append(pred)
        predictions_result.append(pred)
    # extract real value
        obs = test[i]
    # from this moment real value becomes history which will be used as the prediction for next iteration
        history.append(obs)
# Prediction Accuracy
    rmse.append(sqrt(mean_squared_error(test, predictions)))
    count += 1
    
rmse
predictions_result

plt.plot(predictions_result)
plt.plot(log_ret_R[1460:1824])


from statsmodels.tsa.arima_model import ARMA
model = ARMA(log_ret, order=(4, 1)) 
result_arma = model.fit(disp=-1, method='css')
predict_ts=result_arma.predict()

diff_shift_ts = log_ret.shift(1)
diff_recover_1 = predict_ts.add(diff_shift_ts)
log_recover = np.exp(diff_recover_1)
log_recover.dropna(inplace=True)

plt.plot(log_ret[0:730])
plt.plot(predict_ts)
plt.plot(log_recover)
