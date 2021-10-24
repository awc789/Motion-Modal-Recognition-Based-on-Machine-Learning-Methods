import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

from cycle_prepare import *


def read_data(f):
    data = pd.read_csv(f)
    data['date'] = range(len(data))
    data.columns = ['count', 'date']
    # data = data.set_index('date')
    # data.index = pd.to_datetime(data.index)
    ts = data['count']
    # draw_ts(ts)  # 绘图
    return ts

def same_time_data(freq, index, t):
    num = t % freq
    length = len(index)
    t_circle = length // freq
    left_circle = length % freq

    select_list = []
    if left_circle < num:
        for i in range(t_circle):
            var = num + freq * i
            select_list.append(var)
    else:
        for i in range(t_circle + 1):
            var = num + freq * i
            select_list.append(var)

    select_array = []
    for j in range(length):
        if j in select_list:
            select_array.append(True)
        else:
            select_array.append(False)

    select_array = np.array(select_array)

    return select_array




if __name__ == '__main__':
    # file_10 = './data/data_after_kalman/sample10/walk/walk03.txt'
    file_10 = './data/data_after_kalman/sample10/walk/walk03.txt'
    freq, test_size = set_data_10(file_10)

    filename = './data/cycle.txt'
    ts = read_data(filename)

    test = ts[-test_size:]
    train_size = len(ts) - test_size
    train = ts[:len(ts) - test_size]

    ''' ~~~~~~~~~~~~~~~~~~ diff_smooth ~~~~~~~~~~~~~~~~~~ '''
    ''' 差分序列 搜索异常点用前后均值填充 '''
    dif = train.diff().dropna()  # 差分序列
    td = dif.describe()  # 描述性统计得到：min，25%，50%，75%，max值
    high = td['75%'] + 1.5 * (td['75%'] - td['25%'])  # 定义高点阈值，1.5倍四分位距之外
    low = td['25%'] - 1.5 * (td['75%'] - td['25%'])  # 定义低点阈值，同上
    forbid_index = dif[(dif > high) | (dif < low)].index
    i = 0
    while i < len(forbid_index) - 1:
        n = 1  # 发现连续多少个点变化幅度过大，大部分只有单个点
        start = forbid_index[i]  # 异常点的起始索引
        while forbid_index[i + n] == start + 1:
            n += 1
        i += n - 1

        end = forbid_index[i]  # 异常点的结束索引
        # 用前后值的中间值均匀填充
        value = np.linspace(train[start - 1], train[end + 1], n - 1)
        train[start: end] = value
        i += 1

    ''' ~~~~~~~~~~~~~~~~~~ decomp ~~~~~~~~~~~~~~~~~~ '''
    ''' 对时间序列进行分解 '''
    decomposition = seasonal_decompose(train, freq=freq, two_sided=False)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # decomposition.plot()
    # plt.show()

    d = residual.describe()
    delta = d['75%'] - d['25%']
    low_error, high_error = (d['25%'] - 1 * delta, d['75%'] + 1 * delta)

    ''' ~~~~~~~~~~~~~~~~~~ trend_model ~~~~~~~~~~~~~~~~~~ '''
    ''' 为分解出来的趋势数据单独建模 '''
    trend.dropna(inplace=True)
    train_results = sm.tsa.arma_order_select_ic(trend, ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)
    p, q = train_results.bic_min_order[0], train_results.bic_min_order[1]
    # trend_model = sm.tsa.ARIMA(trend, order=(p, 1, q)).fit(disp=-1, method='css')
    trend_model = ARIMA(trend, order=(p, 0, q)).fit(disp=-1, method='css')

    ''' ~~~~~~~~~~~~~~~~~~ predict_new ~~~~~~~~~~~~~~~~~~ '''
    ''' 预测新数据 '''
    n = test_size
    pred_time_index = []
    for i in range(train.index[-1] + 1, train.index[-1] + n + 1):
        pred_time_index.append(i)
    trend_pred = trend_model.forecast(n)[0]

    ''' ~~~~~~~~~~~~~~~~~~ add_season ~~~~~~~~~~~~~~~~~~ '''
    ''' 为预测出的趋势数据添加周期数据和残差数据 '''
    train_season = seasonal
    values = []
    low_conf_values = []
    high_conf_values = []
    for i, t in enumerate(pred_time_index):
        trend_part = trend_pred[i]
        # 相同时间的数据均值

        season_part = train_season[
            same_time_data(freq, train_season.index, t)
        ].mean()
        # 趋势 + 周期 + 误差界限
        predict = trend_part + season_part
        low_bound = trend_part + season_part + low_error
        high_bound = trend_part + season_part + high_error

        values.append(predict)
        low_conf_values.append(low_bound)
        high_conf_values.append(high_bound)

    final_pred = pd.Series(values, index=pred_time_index, name='predict')
    low_conf = pd.Series(low_conf_values, index=pred_time_index, name='low_conf')
    high_conf = pd.Series(high_conf_values, index=pred_time_index, name='high_conf')

    ''' ~~~~~~~~~~~~~~~~~~ 绘图 1  ~~~~~~~~~~~~~~~~~~ '''
    ''' 简单的绘图 '''
    plt.figure(1)
    plt.plot(train)
    plt.plot(final_pred)
    plt.plot(test)
    plt.show()

    ''' ~~~~~~~~~~~~~~~~~~ 绘图 2  ~~~~~~~~~~~~~~~~~~ '''
    ''' 展示预测的准确度 '''
    # plt.subplot(211)
    # plt.plot(train)
    # plt.title('Cycle Predict')
    # plt.subplot(212)
    # final_pred.plot(color='salmon', label='Predict')
    # test.plot(color='steelblue', label='Original')
    # low_conf.plot(color='grey', label='low')
    # high_conf.plot(color='grey', label='high')
    #
    # plt.legend(loc='best')
    # plt.title('RMSE: %.4f' % np.sqrt(sum((final_pred.values - test.values) ** 2) / test.size))
    # plt.tight_layout()
    # plt.show()