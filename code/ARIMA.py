import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools
import get_data


def draw_acf_and_pacf(train):
    fig = plt.figure(figsize=(12, 8))

    # acf
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(train['data'], lags=20, ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout()

    # pacf
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(train['data'], lags=20, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    plt.show()


def draw_bic(train):
    p_min = 0
    d_min = 0
    q_min = 0
    p_max = 5
    d_max = 0
    q_max = 5

    # Initialize a DataFrame to store the results,，以BIC准则
    results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                               columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])

    for p, d, q in itertools.product(range(p_min, p_max + 1),
                                     range(d_min, d_max + 1),
                                     range(q_min, q_max + 1)):
        if p == 0 and d == 0 and q == 0:
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            continue

        try:
            model = sm.tsa.ARIMA(train['data'], order=(p, d, q),
                                 # enforce_stationarity=False,
                                 # enforce_invertibility=False,
                                 )
            results = model.fit()
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
        except:
            continue
    results_bic = results_bic[results_bic.columns].astype(float)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(results_bic,
                     mask=results_bic.isnull(),
                     ax=ax,
                     annot=True,
                     fmt='.2f',
                     )
    ax.set_title('BIC')
    plt.show()


if __name__ == '__main__':
    N = 0  # 选取从 N 开始选取 train 的数据
    N_num = 900  # 选取 train 的数据长度
    t_num = 19  # 选取 test 的数据长度

    file_10 = './data/data_after_kalman/sample10/walk/walk01.txt'

    file_9 = './data/data_after_kalman/sample9/walk/walk01.txt'

    # data 10
    times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_data.get_kalman_data_10(file_10)

    sample = ay
    train = pd.DataFrame(sample[N: N + N_num], columns=['data'])
    test = pd.DataFrame(sample[N + N_num: N + N_num + t_num], columns=['data'])

    # 绘图
    ' train '
    train.plot(figsize=(24, 8))
    plt.legend(bbox_to_anchor=(1, 0.125))
    plt.title('Sensor 10 Data sample ')
    sns.despine()
    ' test '
    # plt.figure(figsize=(24, 8))
    # plt.plot(test['data'], 'g')
    # plt.show()

    # 差分
    train['diff_1'] = train['data'].diff(1)
    train['diff_2'] = train['diff_1'].diff(1)
    # train.plot(subplots=True, figsize=(18, 12))  # 绘图

    train_results = sm.tsa.arma_order_select_ic(train['data'], ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)
    print('AIC', train_results.aic_min_order)
    print('BIC', train_results.bic_min_order)
    p, q = train_results.bic_min_order[0], train_results.bic_min_order[1]

    ' Autocorrelation '
    # model = sm.tsa.ARIMA(train['data'], order=(p, 0, q))
    # results = model.fit()
    # resid = results.resid  # 赋值
    # fig = plt.figure(figsize=(12, 8))
    # fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40)
    # plt.show()

    model = sm.tsa.ARIMA(train['data'], order=(p, 0, q))
    results = model.fit()

    # 修改后的预测数据
    predict_sunspots = results.predict(start=0, end=N_num + t_num + 30, dynamic=False)
    # predict_sunspots = results.predict(start=0, end=N_num + t_num, dynamic=False)

    # predict_sunspots = results.forecast()[0]
    # print(predict_sunspots)
    fig, ax = plt.subplots(figsize=(18, 8))
    ax = (train['data']).plot(ax=ax)
    predict_sunspots.plot(ax=ax)
    plt.plot(range(N_num, len(test['data']) + N_num), test['data'])
    plt.show()
