#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 21:04:17 2019

@author: guisier
"""


import pandas as pd
import numpy as np
from cryptocmd import CmcScraper
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.tsa.api as smt

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
dfdaily['CS']=df['Market Cap']/df['Close']
dfdaily=dfdaily[['Date','Close','CS']]
dfdaily['log_price'] = np.log(dfdaily.Close)
dfdaily['log_ret'] = dfdaily.log_price.diff()
dfdaily=dfdaily.dropna(axis=0,how='any')
log_ret_R=dfdaily['log_ret']
CS=dfdaily['CS']


dfB = scraper1.get_dataframe()
dfdailyB=dfB[::-1]
dfdailyB=dfdailyB[['Date','Close']]
dfdailyB['log_price_B'] = np.log(dfdailyB.Close)
dfdailyB['log_ret_B'] = dfdailyB.log_price_B.diff()
dfdailyB=dfdailyB.dropna(axis=0,how='any')
log_ret_B=dfdailyB['log_ret_B']
CS=dfdaily['CS']


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

# 分解decomposing, make sure it's no trend and seaonality
from statsmodels.tsa.seasonal import seasonal_decompose
res = seasonal_decompose(dfdaily.log_price[0:730], model='additive',freq = 1)
res.plot()
plt.show()

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
        

#白噪声检验,p value 小于0.5 not white noise (只有时间序列不是一个白噪声（纯随机序列）的时候，该序列才可做分析)
from statsmodels.stats.diagnostic import acorr_ljungbox
#返回统计量和p值
noiseRes = acorr_ljungbox(log_ret, lags=1)
print('stat                  | p-value')
for x in noiseRes:
    print(x,'|', end=" ")


import statsmodels.api as sm
plt.figure(figsize=(10,10))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(log_ret, lags=60, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(log_ret, lags=60, ax=ax)
plt.tight_layout()
plt.show()

#according to acf and pacf, candidate model: 
#arma(1,1) arma(4,0) arma(0,1) arima(0,4)

#use order select to decide the order
# statsmodels.tsa.stattools.arma_order_select_ic(y, max_ar=4, max_ma=2, ic='bic', trend='c', model_kw={}, fit_kw={})
from statsmodels.tsa.stattools import arma_order_select_ic
res = arma_order_select_ic(log_ret,max_ar=5,max_ma=5,ic=['aic','bic'],trend='nc')
# res.aic_min_order
res.bic_min_order #(0,1)
res.aic_min_order #(4,0)

# 结果也是（4，0）
best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(log_ret,
                      order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: %6.2f | order: %s'%(best_aic, best_order))


#看下fit train dataset怎么样  (4,0) is better
model = ARMA(log_ret, order=(4,0))
result_arma = model.fit(disp=-1)
plt.plot(log_ret,linewidth='0.5')
plt.plot(result_arma.fittedvalues, color='red',linewidth='0.7')

# use middle 2 years to validate model，choose the best
#ripple data middle 2 years
log_ret_34=log_ret_R[730:1460]


model = ARMA(log_ret_34, order=(0, 4)) 
result_arma = model.fit( disp=-1)
plt.plot(log_ret_34,linewidth='0.5')
plt.plot(result_arma.fittedvalues, color='red',linewidth='0.7')
plt.title('ARMA(0,4)')
plt.show()

model = ARMA(log_ret_34, order=(1, 1)) 
result_arma = model.fit( disp=-1)
plt.plot(log_ret_34,linewidth='0.5')
plt.plot(result_arma.fittedvalues, color='red',linewidth='0.7')
plt.title('ARMA(1,1)')
plt.show()

model = ARMA(log_ret_34, order=(4, 0))
result_AR = model.fit(disp=-1)
plt.plot(log_ret_34,linewidth='0.5')
plt.plot(result_AR.fittedvalues, color='red',linewidth='0.7')
plt.title('ARMA(4,0)')
plt.show()

model = ARMA(log_ret_34, order=(0, 1))
result_AR = model.fit(disp=-1)
plt.plot(log_ret_34,linewidth='0.5')
plt.plot(result_AR.fittedvalues, color='red',linewidth='0.7')
plt.title('ARMA(0,1)')
plt.show()


from math import sqrt
from sklearn.metrics import mean_squared_error
train = list(log_ret)
test = list(log_ret_34)
predictions = []
history = [x for x in train]
for i in range(len(test)):
    # make prediction
    model = ARIMA(history, order=(4,1,0))
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
        predictions = np.append(predictions,pred)
        predictions_result = np.append(predictions_result,pred)
    # extract real value
        obs = test[i]
    # from this moment real value becomes history which will be used as the prediction for next iteration
        history.append(obs)
# Prediction Accuracy
    rmse.append(sqrt(mean_squared_error(test, predictions)))
    count += 1


plt.plot(predictions_result,linewidth='1',color='r',label='prediction')
plt.plot(log_ret_R[1460:1824],linewidth = '0.5',color='b',label='original')
plt.legend()
plt.show()


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
    
    
plt.plot(predictions_result,linewidth='1',color='r',label='prediction')
plt.plot(log_ret_R[1460:1824],linewidth = '0.5',color='b',label='original')
plt.legend()
plt.show()


#看residuals有没有arch effect
model = ARMA(log_ret, order=(4,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
from pandas import DataFrame
from matplotlib import pyplot
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

#there are ARCH effects in your data, 
#because the ACF and PACF values are statistically significantly different from zero.
import statsmodels.api as sm
plt.figure(figsize=(10,10))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(best_mdl.resid**2, lags=60, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(best_mdl.resid**2, lags=60, ax=ax)
plt.tight_layout()
plt.show()

sm.qqplot(best_mdl.resid**2, line='s')

import arch 
from arch import arch_model
am = arch_model(log_ret)
res = am.fit(update_freq=5)
print(res.summary())

model=arch_model(residuls, vol='Garch', p=1, o=0, q=1, dist='Normal')
results=model.fit()
print(results.summary())


#白噪声检验,p value 小于0.5 not white noise (只有时间序列不是一个白噪声（纯随机序列）的时候，该序列才可做分析)
from statsmodels.stats.diagnostic import acorr_ljungbox
#返回统计量和p值
noiseRes = acorr_ljungbox(best_mdl.resid, lags=1)
print('stat                  | p-value')
for x in noiseRes:
    print(x,'|', end=" ")







## circulating supply vs price
fig = plt.figure()
ax3 = fig.add_subplot(111)
ax3.plot(Date, Close,linewidth = '0.5',color='b',label='Ripple')
plt.legend(loc=2)
ax3.tick_params(axis='y', colors='b')
ax3.set_ylabel('Ripple',color='b')

ax3.set_title('cs vs price')
ax4 = ax3.twinx()  # this is the important function
ax4.plot(dfdaily.Date, CS, linewidth = '0.5',color='r',label='CS')
plt.legend(loc=1)
ax4.tick_params(axis='y', colors='r')
ax4.set_ylabel('CS',color='r')
plt.show()
