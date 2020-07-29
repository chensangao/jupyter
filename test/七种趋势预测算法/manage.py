#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error
from math import sqrt

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', '时间序列预测.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    # region 训练数据
    # 选择数据源
    df = pd.read_csv('train.csv', nrows=11856)

    # 将数据集分为训练集和测试集
    train = df[0:10392]
    test = df[10392:]

    # 日期格式转换
    df['Timestamp'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
    df.index = df['Timestamp']
    # 数据按天聚合
    df = df.resample('D').mean()

    train['Timestamp'] = pd.to_datetime(train['Datetime'], format='%d-%m-%Y %H:%M')
    train.index = train['Timestamp']
    train = train.resample('D').mean()

    test['Timestamp'] = pd.to_datetime(test['Datetime'], format='%d-%m-%Y %H:%M')
    test.index = test['Timestamp']
    test = test.resample('D').mean()

    # 设置绘制大小，标题
    train.Count.plot(figsize=(12, 8), title='Daily Ridership', fontsize=140)
    test.Count.plot(figsize=(12, 8), title='Daily Ridership', fontsize=14)
    # endregion

    # region 朴素法
    dd = np.asarray(train['Count'])
    y_hat = test.copy()
    # 取前一时间段数据
    y_hat['naive'] = dd[len(dd) - 1]

    plt.figure(figsize=(12, 8))
    plt.plot(train.index, train['Count'], label='Train')
    plt.plot(test.index, test['Count'], label='Test')
    plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
    plt.legend(loc='best')
    plt.title("朴素法")

    rms = sqrt(mean_squared_error(test['Count'], y_hat['naive']))
    print("朴素法RMS：" + str(rms))
    # endregion

    # region 简单平均法
    y_hat_avg = test.copy()
    # 取训练集的平均值
    y_hat_avg['avg_forecast'] = train['Count'].mean()

    plt.figure(figsize=(12, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
    plt.legend(loc='best')
    plt.title("简单平均法")

    rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['avg_forecast']))
    print("简单平均法RMS：" + str(rms))
    # endregion

    # region 移动平均法
    y_hat_mov_avg = test.copy()
    # 取最近60天的平均值
    y_hat_mov_avg['moving_avg_forecast'] = train['Count'].rolling(60).mean().iloc[-1]

    plt.figure(figsize=(12, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_mov_avg['moving_avg_forecast'], label='Moving Average Forecast')
    plt.legend(loc='best')
    plt.title("移动平均法")

    rms = sqrt(mean_squared_error(test['Count'], y_hat_mov_avg['moving_avg_forecast']))
    print("移动平均法RMS：" + str(rms))
    # endregion

    # region简单指数平滑法
    from statsmodels.tsa.api import SimpleExpSmoothing

    y_hat_exp_sm = test.copy()
    # smoothing_level:权重下降速率
    fit = SimpleExpSmoothing(np.asarray(train['Count'])).fit(smoothing_level=0.6, optimized=False)
    y_hat_exp_sm['SES'] = fit.forecast(len(test))

    plt.figure(figsize=(12, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_exp_sm['SES'], label='SES')
    plt.legend(loc='best')
    plt.title("简单指数平滑法")

    rms = sqrt(mean_squared_error(test['Count'], y_hat_exp_sm['SES']))
    print("简单指数平滑法RMS：" + str(rms))
    # endregion

    # region 霍尔特(Holt)线性趋势法
    import statsmodels.api as sm
    from statsmodels.tsa.api import Holt

    sm.tsa.seasonal_decompose(train['Count']).plot()
    result = sm.tsa.stattools.adfuller(train['Count'])

    y_hat_holt = test.copy()
    fit = Holt(np.asarray(train['Count'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    y_hat_holt['Holt_linear'] = fit.forecast(len(test))

    plt.figure(figsize=(12, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_holt['Holt_linear'], label='Holt_linear')
    plt.legend(loc='best')
    plt.title("Holt线性趋势法")

    rms = sqrt(mean_squared_error(test['Count'], y_hat_holt['Holt_linear']))
    print("霍尔特(Holt)线性趋势法RMS：" + str(rms))
    # endregion

    # region Holt-Winters季节性预测模型
    from statsmodels.tsa.api import ExponentialSmoothing

    y_hat_HoltWinter = test.copy()
    fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=7, trend='add', seasonal='add', ).fit()
    y_hat_HoltWinter['Holt_Winter'] = fit1.forecast(len(test))

    plt.figure(figsize=(12, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_HoltWinter['Holt_Winter'], label='Holt_Winter')
    plt.legend(loc='best')
    plt.title("Holt-Winters季节性预测法")

    rms = sqrt(mean_squared_error(test['Count'], y_hat_HoltWinter['Holt_Winter']))
    print("Holt-Winters季节性预测模型RMS：" + str(rms))
    # endregion

    # region自回归移动平均模型（ARIMA）
    import statsmodels.api as sm

    y_hat_avg = test.copy()
    fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
    y_hat_avg['ARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)

    plt.figure(figsize=(12, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
    plt.legend(loc='best')
    plt.title("ARIMA自回归移动平均法")

    rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['ARIMA']))
    print("自回归移动平均模型（ARIMA）RMS：" + str(rms))
    # endregion

    plt.show()
