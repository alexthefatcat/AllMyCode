# -*- coding: utf-8 -*-
"""Created on Wed May  8 10:10:16 2019@author: milroa1"""

#    Timeseries Prediction Forcasting and Analysis
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots
 





filepath = r"H:\AM\Timeseries__daily-min-temperatures.csv"
df = pd.read_csv(filepath,index_col=0)
df.index = pd.to_datetime(df.index)

df2 =df[1000:2000].copy()

"""
Forecast quality metrics

R squared                     , coefficient of determination (in econometrics it can be interpreted as a percentage of variance explained by the model), (-inf, 1] sklearn.metrics.r2_score 
Mean Absolute Error           , it is an interpretable metric because it has the same unit of measurement as the initial series, [0, +inf) sklearn.metrics.mean_absolute_error
Median Absolute Error         , again an interpretable metric, particularly interesting because it is robust to outliers, [0, +inf)sklearn.metrics.median_absolute_error
Mean Squared Error            , most commonly used, gives higher penalty to big mistakes and vise versa, [0, +inf)sklearn.metrics.mean_squared_error
Mean Squared Logarithmic Error, practically the same as MSE but we initially take logarithm of the series, as a result we give attention to small mistakes as well, usually is used when data has exponential trends, [0, +inf)sklearn.metrics.mean_squared_log_error
Mean Absolute Percentage Error, same as MAE but percentage, — very convenient when you want to explain the quality of the model to your management, [0, +inf), not implemented in sklearn


#Before we start modeling we should mention such an important property of time series as stationarity.
#If the process is stationary that means it doesn’t change 

# A timeseries is stationarity if its statistical properties over time, namely mean and variance,covaraince do not change over time 
# these methods are for stationarity
    Moving Averages             : df.rolling(24).mean()
    Weighted Moving Averages    : df.rolling(5).apply(lambda x: sum(w*v for w,v in zip([0.05,0.05,0.1,0.2,0.6],x)))
    Exponential smoothing       : sort of smothed drifted average + value
    Double Expotenial smoothing : includes trend
    Triple exponential smoothing a.k.a. Holt-Winters : includes seasonlity # not stock market, think tempature

for non-stationarity
   one example could be random walk
   however to work with random walk the differnce x(t) -x(t-1) is stationarity
   so by the Dick Fuller test this distritbtion has a Intergrated order 1 can be higher if you need to take more differetnails
   
We can fight non-stationarity using different approaches — various order differences, trend and seasonality removal, 
smoothing, also using transformations like Box-Cox or logarithmic.   
   
ARIMA: AutoRegressive Integrated Moving Average   #  SARIMA (seasonal ARIMA) 
    In an ARIMA model we transform a time series into stationary one(series without trend or seasonality) using differencing
        Auto-regresive: uses regression of past values
                   Yt = B0 + B1Yt-1 + B2Yt-2  # do this for the time points
        MA: Moving averages
        p,d,q #params (0,0,1):Moving Average,
        p number of times the differntiate the series

    Autoregressive Integrated Moving Average or ARIMA(p,d,q)model of Y.p is the number of lagged values of Y∗which represents the autoregressive (AR)
     nature of model, q is the number of lagged values of the error term which represents the moving average (MA) nature of model and dis the number 
     of times Y has to be differences to produce the stationary Y∗. The term integrated implies that in order to obtain a forecast of Y, we have to
     sum up (or integrate over) the values of Y∗ because Y∗are the differenced values of the original series Y.If no differencing is involved, this 
     model is called an Autoregressive Moving Average [ARMA(p,q)] 


Expotenial smoothing quit simple good results, not state of the art
Holt-Winters
ARIMA these two are about even

#super state of the art
LSTM
Boosted

# maybe come up with some autoencoder-LSTM with some transfer learning


"""
#from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
#from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
#
#def mean_absolute_percentage_error(y_true, y_pred): 
#    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#
#

#df.plot()
#def moving_average(series, n):
#    return np.average(series[-n:])
#moving_average(df["Temp"], 24)
 
# moving averages are a good form of prediction this will smooth the data








df2.plot()


def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "yo", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

# Moving Averages in Pandas
df2["moving_average_24"]=df2["Temp"].rolling(24).mean()

plotMovingAverage(df2[["Temp"]], 4, plot_intervals=True,plot_anomalies=True)


# Moving Averages use a window of the last few days to predict next
# weighted average wieghts more recent data more
    # could use a wieghted window
zscore = lambda x: (x - x.mean()) / x.std()
tmp.rolling(5).apply(zscore)

# Exponential smoothing
result = [series[0]] # first value is same as series
for n in range(1, len(series)):
    result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    
# Double exponential smoothing    
#       Series decomposition should help us — we obtain two components: intercept (also, level) ℓ and trend (also, slope) b. 
    
def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

#  Triple exponential smoothing a.k.a. Holt-Winters
#    If the data is seasonal

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result








from xgboost import XGBRegressor 

xgb = XGBRegressor()
xgb.fit(X_train_scaled, y_train)

plotModelResults(xgb,X_train=X_train_scaled, X_test=X_test_scaled,plot_intervals=True, plot_anomalies=True)








