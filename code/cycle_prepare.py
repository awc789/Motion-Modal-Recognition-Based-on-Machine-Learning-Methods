import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from get_data import get_kalman_data_10, get_kalman_data_9
from wave_detect import wave_double
from wave_filter import wave_mix
from setting import get_thh

def get_freq(N, N_num, ax, ay, az, wx, wy, wz):
    temp_ax = ax[N: N + N_num]
    temp_ay = ay[N: N + N_num]
    temp_az = az[N: N + N_num]
    temp_wx = wx[N: N + N_num]
    temp_wy = wy[N: N + N_num]
    temp_wz = wz[N: N + N_num]
    # plt.plot(temp_wy)

    thh = 1000
    mix = wave_mix(temp_ax, temp_ay, temp_az, temp_wx, temp_wy, temp_wz) # 平滑滤波
    waves = wave_double(mix, thh)
    thh = get_thh(mix, waves) # 自适应thh
    waves = wave_double(mix, thh)

    freq = []
    for i in range(len(waves)):
        p1 = waves[i][0]
        p3 = waves[i][2]
        d = p3 - p1
        freq.append(d)

    freq = int(np.mean(freq))
    return waves, freq


def set_data_9(file_9):
    # N = 15000  # 选取从 N 开始选取 train 的数据
    N_num = 3000  # 选取 train 的数据长度
    # t_num = 500  # 选取 test 的数据长度

    # file_9 = './data/data_after_kalman/sample9/walk/walk03.txt'
    times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3 = get_kalman_data_9(file_9)
    N = random.randint(N_num * 2, len(times) - 2 * N_num)
    # 选取样本的训练和测试数据
    sample = wy
    # train = pd.DataFrame(sample[N: N + N_num], columns=['data'])
    # test = pd.DataFrame(sample[N + N_num: N + N_num + t_num], columns=['data'])
    # 获得周期
    waves, freq = get_freq(N, N_num, ax, ay, az, wx, wy, wz)
    print('freq: ', freq)
    start_num = waves[0][0]
    end_num = waves[-1][2]
    test_size = waves[-1][2] - waves[-2][0]
    print('test_size: ', test_size)
    np.savetxt('./data/cycle.txt', sample[N + start_num: N + end_num], fmt="%f", delimiter=',')

    return freq, test_size


def set_data_10(file_10):
    # N = 15000  # 选取从 N 开始选取 train 的数据
    N_num = 3000  # 选取 train 的数据长度
    # t_num = 500  # 选取 test 的数据长度

    # file_10 = './data/data_after_kalman/sample10/walk/walk03.txt'
    times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_kalman_data_10(file_10)
    N = random.randint(N_num * 2, len(times) - 2 * N_num)
    # 选取样本的训练和测试数据
    sample = wy
    # train = pd.DataFrame(sample[N: N + N_num], columns=['data'])
    # test = pd.DataFrame(sample[N + N_num: N + N_num + t_num], columns=['data'])
    # 获得周期
    waves, freq = get_freq(N, N_num, ax, ay, az, wx, wy, wz)
    print('freq: ', freq)
    start_num = waves[0][0]
    end_num = waves[-1][2]
    test_size = waves[-1][2] - waves[-2][0]
    print('test_size: ', test_size)
    np.savetxt('./data/cycle.txt', sample[N + start_num: N + end_num], fmt="%f", delimiter=',')

    # plt.plot(sample[start_num: end_num])

    return freq, test_size
