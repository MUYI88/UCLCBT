#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:29:06 2019

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
from pandas import DataFrame
from matplotlib import pyplot
import arch 
from arch import arch_model
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error
import pyflux as pf
from math import sqrt
import pyflux as pf



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

#ripple 用bitcoin来定价的 log return,看作一个新的time series
excess_ret=log_ret_R-log_ret_B
np.mean(excess_ret)
np.std(excess_ret)
np.min(excess_ret)
np.max(excess_ret)
excess_ret.skew(axis=0)
excess_ret.kurt(axis=0)


#用1，2年的ripple log return - botcoin log return 进行fit
train=[]
train[0:730]=excess_ret[0:730]


# 分解decomposing, make sure it's no trend and seaonality
from statsmodels.tsa.seasonal import seasonal_decompose
res = seasonal_decompose(train, model='additive',freq = 1)
res.plot()
plt.show()

#log return is stationary
from statsmodels.tsa.stattools import adfuller
result = adfuller(train, autolag='AIC')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# We can see that our statistic value of -11.935466 is less than the value of -3.439 
#at 1% or 5% or 10%. This suggests that we can reject the null hypothesis with 
#a significance level of less than 1% or 5% or 10% 
#Rejecting the null hypothesis means that the process has no unit root, 
#and in turn that the time series is stationary or does not have time-dependent structure.

#白噪声检验,p value 小于0.5 not white noise (只有时间序列不是一个白噪声（纯随机序列）的时候，该序列才可做分析)
from statsmodels.stats.diagnostic import acorr_ljungbox
#返回统计量和p值
noiseRes = acorr_ljungbox(train, lags=1)
print('stat                  | p-value')
for x in noiseRes:
    print(x,'|', end=" ")


import statsmodels.api as sm
plt.figure(figsize=(10,10))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(train, lags=60, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(train, lags=60, ax=ax)
plt.tight_layout()
plt.show()

#according to acf and pacf, candidate model: 
#arma(1,1) arma(4,0) arma(0,1) arima(0,4)

#use order select to decide the order
# statsmodels.tsa.stattools.arma_order_select_ic(y, max_ar=4, max_ma=2, ic='bic', trend='c', model_kw={}, fit_kw={})
from statsmodels.tsa.stattools import arma_order_select_ic
res = arma_order_select_ic(train,ic=['aic','bic'],trend='nc')
# res.aic_min_order
res.bic_min_order #(0,1)
res.aic_min_order #(4,1)

# 结果是（4,1）            
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

# 首先试着用arma（4，1）来model mean 
model = ARMA(train, order=(4,1))
model_fit = model.fit(disp=-1)
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

DataFrame(error).plot(kind='kde')
pyplot.show()
print(DataFrame(error).describe())


#序列进行混成检验（Ljung-Box）原假设H0:序列没有相关性，备择假设H1:序列具有相关性
m = 25 # 我们检验25个自相关系数
acf,q,p = sm.tsa.acf(errorsq,nlags=m,qstat=True)  ## 计算自相关系数 及p-value
out = np.c_[range(1,26), acf[1:], q, p]
output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
output = output.set_index('lag')
output   #p-value小于显著性水平0.05，我们拒绝原假设，即认为序列具有相关性。因此具有ARCH效应。

# 看下square residual序列的acf和pacf
import statsmodels.api as sm
plt.figure(figsize=(10,10))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(errorsq, lags=60, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(errorsq, lags=60, ax=ax)
plt.tight_layout()
plt.show()


# 用arma（4，1）arch（3）validate 3，4年
validate=[]
validate[0:730]=excess_ret[730:1460]

am = arch.arch_model(error,mean='zero',lags=0,vol='ARCH',p=3) 
res = am.fit()
print(res.summary())

model = ARMA(train, order=(4,1))
model_fit = model.fit(disp=-1)
print(model_fit.summary())


coef1 = np.array([0.0818,0.0360,-0.1881,0.7529])
miu1 = 0.0006
coef2 = -0.5439
alpha = np.array([0.0993,0.0936,0.4078])
omega1 = 0.0011243


y = train[726:730]
err=np.array([0.0019030288658085596])
errorsq=error**2
epsilonsq=errorsq[727:730]
p=4
q1=1
q2=3
for i in range(0,730):
    mean = np.sum(y[i:(i+p)]*coef1)+np.sum(err[i:(i+q1)]*coef2)+miu1
    sigmasq = np.sum(epsilonsq[i:(i+q2)]*alpha)+omega1
    epsilon_t = np.random.normal(loc=0,scale = sqrt(sigmasq))
    y = np.append(y,mean+epsilon_t)
    epsilonsq = np.append(epsilonsq, epsilon_t*epsilon_t)
    err=np.append(err, epsilon_t)

plt.plot(y[4:],color='red')
plt.plot(validate,linewidth='0.7')

# 用arma（1，1）arch（3）validate 3，4年
model = ARMA(train, order=(1,1))
model_fit = model.fit(disp=-1)
plt.plot(train,linewidth='0.7')
plt.plot(model_fit.fittedvalues, color='red',linewidth='0.7')

error = train -  model_fit.fittedvalues
errorsq = np.square(error)

validate=[]
validate[0:730]=excess_ret[730:1460]

am = arch.arch_model(error,mean='zero',lags=0,vol='ARCH',p=3) 
res = am.fit()
print(res.summary())

model = ARMA(train, order=(1,1))
model_fit = model.fit(disp=-1)
print(model_fit.summary())


coef1 = np.array([-0.0563])
miu1 = 0.0007
coef2 = 0.2719
alpha = np.array([0.0905,0.0923,0.4182])
omega1 = 0.0011128


y = train[729:730]
err=np.array([-0.00157639112])
errorsq=error**2
epsilonsq=errorsq[727:730]
p=1
q1=1
q2=3
for i in range(0,730):
    mean = np.sum(y[i:(i+p)]*coef1)+np.sum(err[i:(i+q1)]*coef2)+miu1
    sigmasq = np.sum(epsilonsq[i:(i+q2)]*alpha)+omega1
    epsilon_t = np.random.normal(loc=0,scale = sqrt(sigmasq))
    y = np.append(y,mean+epsilon_t)
    epsilonsq = np.append(epsilonsq, epsilon_t*epsilon_t)
    err=np.append(err, epsilon_t)

plt.plot(y[1:],color='red')
plt.plot(validate,linewidth='0.7')



# 用arma（4，1）arch（10）validate 3，4年
validate=[]
validate[0:730]=excess_ret[730:1460]

model = ARMA(train, order=(4,1))
model_fit = model.fit(disp=-1)
print(model_fit.summary())

am = arch.arch_model(error,mean='zero',lags=0,vol='ARCH',p=10) 
res = am.fit()
print(res.summary())

coef1 = np.array([0.0818,0.0360,-0.1881,0.7529])
miu1 = 0.0006
coef2 = -0.5439
alpha = np.array([0.0808,0.0109,0.00057931,0.00081997,0.0186,0.0211,0.0124,0.0589,0.0812,0.3772])
omega1 = 0.00093235


y = train[726:730]
err=np.array([0.0019030288658085596])
errorsq=error**2
epsilonsq=errorsq[720:730]
p=4
q1=1
q2=10
for i in range(0,730):
    mean = np.sum(y[i:(i+p)]*coef1)+np.sum(err[i:(i+q1)]*coef2)+miu1
    sigmasq = np.sum(epsilonsq[i:(i+q2)]*alpha)+omega1
    epsilon_t = np.random.normal(loc=0,scale = sqrt(sigmasq))
    y = np.append(y,mean+epsilon_t)
    epsilonsq = np.append(epsilonsq, epsilon_t*epsilon_t)
    err=np.append(err, epsilon_t)

plt.plot(y[4:],color='red')
plt.plot(validate,linewidth='0.7')


# 用arma（4，1）arch（19）validate 3，4年
validate=[]
validate[0:730]=excess_ret[730:1460]

model = ARMA(train, order=(4,1))
model_fit = model.fit(disp=-1)
print(model_fit.summary())

am = arch.arch_model(error,mean='zero',lags=0,vol='ARCH',p=19) 
res = am.fit()
print(res.summary())

coef1 = np.array([0.0818,0.0360,-0.1881,0.7529])
miu1 = 0.0006
coef2 = -0.5439
alpha = np.array([0,0,0,0.0153,0,0,0,0.0531,0.0174,0.0727,0,0,0,0.0285,0.0213,0.0028224,0.1068,0.0812,0.3405])
omega1 = 0.00076513


y = train[726:730]
err=np.array([0.0019030288658085596])
errorsq=error**2
epsilonsq=errorsq[711:730]
p=4
q1=1
q2=19
for i in range(0,730):
    mean = np.sum(y[i:(i+p)]*coef1)+np.sum(err[i:(i+q1)]*coef2)+miu1
    sigmasq = np.sum(epsilonsq[i:(i+q2)]*alpha)+omega1
    epsilon_t = np.random.normal(loc=0,scale = sqrt(sigmasq))
    y = np.append(y,mean+epsilon_t)
    epsilonsq = np.append(epsilonsq, epsilon_t*epsilon_t)
    err=np.append(err, epsilon_t)

plt.plot(y[4:],color='red',linewidth='0.9')
plt.plot(validate,linewidth='0.7')



#choose arma(0,1)-arch(3) to forecast year 5
train=[]
train[0:1460]=excess_ret[0:1460]


model = ARMA(train, order=(0,1))
model_fit = model.fit(disp=-1)
print(model_fit.summary())

error = train -  model_fit.fittedvalues


am = arch.arch_model(error,mean='zero',lags=0,vol='ARCH',p=3) 
res = am.fit()
print(res.summary())


coef1 = 0
miu1 = 0.0017
coef2 = 0.0448
alpha = np.array([0.1216,0.2541,0.5257])
omega1 = 0.0013108

y = []
err=np.array([-0.0029087])
errorsq=error**2
epsilonsq=errorsq[727:730]
p=0
q1=1
q2=3
for i in range(0,365):
    mean = np.sum(y[i:(i+p)]*coef1)+np.sum(err[i:(i+q1)]*coef2)+miu1
    sigmasq = np.sum(epsilonsq[i:(i+q2)]*alpha)+omega1
    epsilon_t = np.random.normal(loc=0,scale = sqrt(sigmasq))
    y = np.append(y,mean+epsilon_t)
    epsilonsq = np.append(epsilonsq, epsilon_t*epsilon_t)
    err=np.append(err, epsilon_t)

plt.plot(y,color='red',linewidth='0.9')
plt.plot(excess_ret[1460:],linewidth='0.7')








# correlation
c= Close.rolling(90).corr(CloseB)
plt.plot(c)

d= Close.rolling(90).std()
e= CloseB.rolling(90).std()
plt.plot(d)
plt.plot(e,color='red')

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(d,linewidth = '0.5',color='b',label='Ripple')
plt.legend(loc=2)
ax1.tick_params(axis='y', colors='b')
ax1.set_ylabel('Ripple',color='b')

ax1.set_title(' volatility')
ax2 = ax1.twinx()  # this is the important function
ax2.plot(e, linewidth = '0.5',color='r',label='Bitcoin')
plt.legend(loc=1)
ax2.tick_params(axis='y', colors='r')
ax2.set_ylabel('Bitcoin',color='r')

ax1 = fig.add_subplot(212)
c= Close.rolling(90).corr(CloseB)
plt.plot(c)

plt.show()


am = arch.arch_model(train,mean='AR',lags=1,vol='ARCH',p=2) 
res = am.fit()
res.params
res.plot()
plt.plot(train)
res.hedgehog_plot()

pre = res.forecast(horizon=10,start=478).iloc[478]
plt.figure(figsize=(10,4))
plt.plot(test,label='realValue')
pre.plot(label='predictValue')
plt.plot(np.zeros(10),label='zero')
plt.legend(loc=0)