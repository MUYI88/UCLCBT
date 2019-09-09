#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:44:32 2019

@author: guisier
"""


from IPython import get_ipython;   
get_ipython().magic('reset -sf')


import pandas as pd
import numpy as np
from cryptocmd import CmcScraper
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.tsa.api as smt
from pandas import DataFrame
from matplotlib import pyplot
import arch 
from arch import arch_model
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error
import pyflux as pf
from math import sqrt
import pyflux as pf
import matplotlib.pyplot as plt
import seaborn as sns


#ripple data 5 years
# initialise scraper without time interval
scraper = CmcScraper("XRP","01-06-2014", "02-06-2019")
# get raw data as list of list
headers, data = scraper.get_data()
# get data in a json format
xrp_json_data = scraper.get_data("json")
# export the data as csv file, you can also pass optional `name` parameter
scraper.export("csv", name="xrp_all_time")
# Pandas dataFrame for the same data
df = scraper.get_dataframe()

#bitcoin data 5 years
scraper1 = CmcScraper("BTC","01-06-2014", "02-06-2019")
# get raw data as list of list
headers, data = scraper1.get_data()
# get data in a json format
xrp_json_data = scraper1.get_data("json")
# export the data as csv file, you can also pass optional `name` parameter
scraper1.export("csv", name="btc_all_time")
# Pandas dataFrame for the same data
dfB = scraper1.get_dataframe()

# daily price (figure 3.1) 
Date=df['Date']
Close=df['Close']
DateB=dfB['Date']
CloseB=dfB['Close']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(Date, Close,linewidth = '0.5',color='b',label='XRP')
plt.legend(loc=2)
ax1.tick_params(axis='y', colors='b')
ax1.set_ylabel('XRP',color='b')

ax1.set_title('Daily closing price $')
ax2 = ax1.twinx()  # this is the important function
ax2.plot(DateB, CloseB, linewidth = '0.5',color='r',label='BTC')
plt.legend(loc=1)
ax2.tick_params(axis='y', colors='r')
ax2.set_ylabel('BTC',color='r')
plt.show()


# daily log return 
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

# weekly log return 
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

#use BTC to price XRP
excess_ret=log_ret_R-log_ret_B

# figure 3.2, three subplots together
plt.figure(figsize=(10,10))
plt.subplot(311)
plt.title('Daily log-return')
plt.plot(Date,log_ret_R,linewidth = '0.5',color='b',label='XRP',alpha=0.5)
plt.plot(DateB,log_ret_B,linewidth = '0.5',color='r',label='BTC',alpha=0.7)
plt.legend()
plt.subplot(312)
plt.title('Weekly log-return')
plt.plot(DateWR,log_ret_weekR,linewidth = '0.5',color='b',label='XRP')
plt.plot(DateWB,log_ret_weekB,linewidth = '0.5',color='r',label='BTC')
plt.legend()
plt.subplot(313)
plt.title('XRP log-return minus BTC log-return')
plt.plot(Date,excess_ret,linewidth = '0.5',color='b',label='R-B')
plt.legend()
plt.tight_layout()
plt.show()

# statistical Properties for 5 years daily log-returns  (table 3.1)
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


np.mean(excess_ret)
np.std(excess_ret)
np.min(excess_ret)
np.max(excess_ret)
excess_ret.skew(axis=0)
excess_ret.kurt(axis=0)

#histograms (figure 3.3)
sns.distplot(excess_ret, hist=True, kde=True, 
             bins=int(180/1), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})
plt.title('Density plot and Histogram of R-B')


sns.distplot(log_ret_R, hist=True, kde=True, 
             bins=int(180/1), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})
plt.xlim(-0.75, 1.1)
plt.title('Density plot and Histogram of XRP')


sns.distplot(log_ret_B, hist=True, kde=True, 
             bins=int(180/1), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})
plt.xlim(-0.75, 1.1)
plt.title('Density plot and Histogram of BTC')


# in-sample data 
train=[]
train[0:728]=excess_ret[0:728]


# decomposing, make sure it's no trend and seaonality (figure 3.4)
from statsmodels.tsa.seasonal import seasonal_decompose
res = seasonal_decompose(train, model='additive',freq = 1)
res.plot()
plt.show()

# check the stationarity (Table 3.2)
# ADF
from statsmodels.tsa.stattools import adfuller
result = adfuller(train, autolag='AIC')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# We can see that our statistic value of -11.895804 is less than the value of -3.439 
#at 1% or 5% or 10%. This suggests that we can reject the null hypothesis with 
#a significance level of less than 1% or 5% or 10% 
#Rejecting the null hypothesis means that the process has no unit root, 
#and in turn that the time series is stationary or does not have time-dependent structure.
#define function for ADF test

# ADF 
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(train, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
adf_test(train)

# KPSS
from statsmodels.tsa.stattools import kpss
kpsstest = kpss(train, regression='c')
kpsstest

# white noise test
# p value less than 0.5,not white noise 
from statsmodels.stats.diagnostic import acorr_ljungbox
#返回统计量和p值
noiseRes = acorr_ljungbox(train, lags=1)
print('stat                  | p-value')
for x in noiseRes:
    print(x,'|', end=" ")


# ACF abd PACF plots for in-sample data (figure 4.1)
import statsmodels.api as sm
plt.figure(figsize=(10,10))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(train, lags=60, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(train, lags=60, ax=ax)
plt.tight_layout()
plt.show()

#according to acf and pacf, candidate model: 
#arma(1,1) arma(4,1) arma(0,1) arima(0,4)

#use order select to decide the order
# statsmodels.tsa.stattools.arma_order_select_ic(y, max_ar=4, max_ma=2, ic='bic', trend='c', model_kw={}, fit_kw={})
from statsmodels.tsa.stattools import arma_order_select_ic
res = arma_order_select_ic(train,ic=['aic','bic'],trend='nc')
# res.aic_min_order
res.bic_min_order #(0,1)
res.aic_min_order #(4,1)

# another way, also (4,1)           
best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5)
d_rng = range(3)
for i in rng:
         for j in rng:
             for d in d_rng:
                try:
                 tmp_mdl = smt.ARIMA(train,
                      order=(i,d,j)).fit(method='mle', trend='nc')
                 tmp_aic = tmp_mdl.aic
                 if tmp_aic < best_aic:
                     best_aic = tmp_aic
                     best_order = (i,d,j)
                     best_mdl = tmp_mdl
                except: continue
                        
print('aic: %6.2f | order: %s'%(best_aic, best_order))


# first try ARMA(0,4) to model mean, error square has correlation so still need garch model, 
# parameters can change, like ARMA(0,1), ARMA(4,1)  

model = ARMA(train, order=(0,4))
model_fit = model.fit(disp=-1)
print(model_fit.summary())
plt.plot(train,linewidth='0.7')
plt.plot(model_fit.fittedvalues, color='red',linewidth='0.7')

error = train -  model_fit.fittedvalues
errorsq = np.square(error)
plt.figure(figsize=(10,6))
plt.subplot(211)
plt.plot(error,label = 'residual')
plt.legend()
plt.subplot(212)
plt.plot(errorsq,label='squared residual')
plt.legend(loc=0)

# Table 4.1
DataFrame(error).plot(kind='kde')
pyplot.show()
print(DataFrame(error).describe())

# ACF and PACF plots of residual squared for different ARMA models (figure 4.2)
import statsmodels.api as sm
plt.figure(figsize=(10,10))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(errorsq, lags=60, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(errorsq, lags=60, ax=ax)
plt.tight_layout()
plt.show()


