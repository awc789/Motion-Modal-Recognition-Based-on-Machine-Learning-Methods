import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from get_data import get_kalman_data_10, get_kalman_data_9
from wave_detect import wave_double
from wave_filter import wave_mix
from setting import get_thh
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


def set_data_9(file_9, file_10):
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
    # 保存9轴数据信息
    np.savetxt('./data/cycle_9.txt', sample[N + start_num: N + end_num], fmt="%f",
               delimiter=',')

    # plt.plot(sample[N+start_num: N+end_num])

    ''' 方案 1 '''
    # # 获得10轴传感器的相应数据
    # count_9 = setting.calculate_times_count(times)  # 将10轴数据进行转换
    # times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_kalman_data_10(file_10) # 读取数据
    # count_10 = setting.calculate_times_count(times)  # 将9轴数据进行转换
    # # 9轴数据和10轴数据配对
    # new_start_num = setting.judge_transform(count_10, count_9[N + start_num])
    # new_end_num = setting.judge_transform(count_10, count_9[N + end_num])
    # new_start_num = count_10.index(new_start_num)
    # new_end_num = count_10.index(new_end_num)
    # # 保存9轴数据信息
    # sample = wy
    # np.savetxt('./data/cycle_10.txt', sample[new_start_num: new_end_num], fmt="%f",
    #            delimiter=',')

    ''' 方案 2 '''
    times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_kalman_data_10(file_10)  # 读取数据
    # 保存9轴数据信息
    sample = wy
    np.savetxt('./data/cycle_10.txt', sample[N + start_num: N + end_num], fmt="%f",
               delimiter=',')

    return freq, waves, test_size

def set_data_10(file_9, file_10):
    # 选取 train 的数据长度
    N_num = 8000
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

    ''' 方案 1 '''
    # # 获得9轴传感器的相应数据
    # count_10 = setting.calculate_times_count(times) # 将10轴数据进行转换
    # times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3 = get_kalman_data_9(file_9) # 读取数据
    # count_9 = setting.calculate_times_count(times) # 将9轴数据进行转换
    # # 9轴数据和10轴数据配对
    # new_start_num = setting.judge_transform(count_9, count_10[N + start_num])
    # new_end_num = setting.judge_transform(count_9, count_10[N + end_num])
    # new_start_num = count_9.index(new_start_num)
    # new_end_num = count_9.index(new_end_num)
    # # 保存9轴数据信息
    # sample = wy
    # np.savetxt('./data/cycle_9.txt', sample[new_start_num: new_end_num], fmt="%f",
    #            delimiter=',')

    ''' 方案 2 '''
    times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3 = get_kalman_data_9(file_9)  # 读取数据
    # 保存9轴数据信息
    sample = wy
    np.savetxt('./data/cycle_9.txt', sample[N + start_num: N + end_num], fmt="%f",
               delimiter=',')

    return freq, waves, test_size
