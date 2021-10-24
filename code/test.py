import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from rnn_prepare import get_freq
from get_data import get_kalman_data_10, get_kalman_data_9
import setting

def read_data(f):
    data = pd.read_csv(f)
    data['date'] = range(len(data))
    data.columns = ['count', 'date']
    # data = data.set_index('date')
    # data.index = pd.to_datetime(data.index)
    ts = data['count']
    # draw_ts(ts)  # 绘图
    return ts


def set_data_9(file_9):
    # 选取 train 的数据长度
    N_num = 3000
    # 读取数据
    times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3 = get_kalman_data_9(file_9)

    # 选择合适的数据开始位置
    if N_num * 2 > len(times) - 2 * N_num:
        N = random.randint(N_num * 2, len(times) - 2 * N_num)
    else:
        N = random.randint(N_num, len(times) - N_num)

    # 选取样本的训练和测试数据
    sample = wy
    # 获得周期
    waves, freq = get_freq(N, N_num, ax, ay, az, wx, wy, wz)
    print('freq: ', freq)
    start_num = waves[0][0]
    end_num = waves[-1][2]
    test_size = waves[-1][2] - waves[-2][0]
    print('test_size: ', test_size)
    np.savetxt('./data/cycle.txt', sample[N + start_num: N + end_num], fmt="%f",
               delimiter=',')

    plt.plot(sample[N+start_num: N+end_num])

    return freq, test_size


def set_data_10(file_9, file_10):
    # 选取 train 的数据长度
    N_num = 3000
    # 读取数据
    times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_kalman_data_10(file_10)

    # 选择合适的数据开始位置
    if N_num * 2 > len(times) - 2 * N_num:
        N = random.randint(N_num * 2, len(times) - 2 * N_num)
    else:
        N = random.randint(N_num, len(times) - N_num)

    # 选取样本的训练和测试数据
    sample = wy
    # 获得周期
    waves, freq = get_freq(N, N_num, ax, ay, az, wx, wy, wz)
    print('freq: ', freq)
    start_num = waves[0][0]
    end_num = waves[-1][2]
    test_size = waves[-1][2] - waves[-2][0]
    print('test_size: ', test_size)
    # 保存10轴数据信息
    np.savetxt('./data/cycle_10.txt', sample[N + start_num: N + end_num], fmt="%f",
               delimiter=',')

    # plt.plot(sample[N+start_num: N+end_num])

    times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3 = get_kalman_data_9(file_9)  # 读取数据
    # 保存9轴数据信息
    sample = wy
    np.savetxt('./data/cycle_9.txt', sample[N + start_num: N + end_num], fmt="%f",
               delimiter=',')


    return freq, waves, test_size


if __name__ == '__main__':
    file_9 = './data/data_after_kalman/sample9/walk/walk01.txt'
    file_10 = './data/data_after_kalman/sample10/walk/walk01.txt'

    freq, waves, test_size = set_data_10(file_9, file_10)

    filename_9 = './data/cycle_9.txt'
    filename_10 = './data/cycle_10.txt'
    ts_9 = read_data(filename_9).values
    ts_10 = read_data(filename_10).values

    plt.subplot(211)
    plt.plot(ts_9)
    plt.subplot(212)
    plt.plot(ts_10)

    print(len(ts_9), len(ts_10))