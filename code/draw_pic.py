import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_data import get_kalman_data_10, get_data_10
from wave_filter import wave_mix
from wave_detect import wave_double
from setting import get_thh
from wave_filter import after_kalman
from get_feature import fourier_transform


if __name__ == '__main__':

    file = './data/sample10/run/sample10_run02.txt'
    times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_data_10(file)
    rows = len(times)

    thh = 1000
    mix = wave_mix(ax, ay, az, wx, wy, wz)  # 平滑滤波
    waves = wave_double(mix, thh)
    thh = get_thh(mix, waves)  # 自适应thh
    waves = wave_double(mix, thh)

    p1 = waves[26][0]
    p3 = waves[29][2]

    file = './data/data_after_kalman/sample10/run/run02.txt'
    times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_kalman_data_10(file)

    # ax_walk = ax[waves[100][0]: waves[100][2]]
    #
    # # file = './data/data_after_kalman/sample10/run/run02.txt'
    # times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_kalman_data_10(file)
    #
    # thh = 1000
    # mix = wave_mix(ax, ay, az, wx, wy, wz)  # 平滑滤波
    # waves = wave_double(mix, thh)
    # thh = get_thh(mix, waves)  # 自适应thh
    # waves = wave_double(mix, thh)
    #
    # ax_run = ax[waves[20][0]: waves[20][2]]
    #
    # file = './data/data_after_kalman/sample10/upstairs/upstairs01.txt'
    # times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_kalman_data_10(file)
    #
    # thh = 1000
    # mix = wave_mix(ax, ay, az, wx, wy, wz)  # 平滑滤波
    # waves = wave_double(mix, thh)
    # thh = get_thh(mix, waves)  # 自适应thh
    # waves = wave_double(mix, thh)
    #
    # ax_upstairs = ax[waves[50][0]: waves[50][2]]
    #
    # file = './data/data_after_kalman/sample10/bicycle/bicycle04.txt'
    # times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_kalman_data_10(file)
    #
    # thh = 1000
    # mix = wave_mix(ax, ay, az, wx, wy, wz)  # 平滑滤波
    # waves = wave_double(mix, thh)
    # thh = get_thh(mix, waves)  # 自适应thh
    # waves = wave_double(mix, thh)
    #
    # ax_bicycle = ax[waves[100][0]: waves[100][2]]
    #
    #
    #
    # plt.figure(figsize= (6,6))
    #
    # plt.subplot(221)
    # plt.title('Walk')
    # plt.plot(ax_walk)
    # # plt.xticks([])
    # plt.yticks([])
    #
    # plt.subplot(222)
    # plt.title('Run')
    # plt.plot(ax_run)
    # # plt.xticks([])
    # plt.yticks([])
    #
    # plt.subplot(223)
    # plt.title('Upstairs')
    # plt.plot(ax_upstairs)
    # # plt.xticks([])
    # plt.yticks([])
    #
    # plt.subplot(224)
    # plt.title('Bicycle')
    # plt.plot(ax_bicycle)
    # # plt.xticks([])
    # plt.yticks([])
    #
    # plt.show()

    # num = 3
    # ax_new =ax[0:num]
    # for i in range(num, len(ax)):  # 9 --- rows-1
    #     sum_1 = 0
    #     for j in range(num + 1):
    #         sum_1 = sum_1 + ax[i + j - num]  # 每10个求和
    #     avg_1 = sum_1 / (num + 1)
    #     ax_new.append(avg_1)

    # af_ax, af_ay, af_az, af_wx, af_wy, af_wz = after_kalman(ax_new, ay, az, wx, wy, wz, rows)


    sample_f = wy[p1:p3]

    p_new = np.linspace(p1, p3, 100)
    ax_fourier = fourier_transform(sample_f, p_new)

    N = len(p_new)
    length = np.arange(N)
    half_length = length[range(int(N / 2))]

    plt.figure(figsize=(5, 4))
    plt.plot(half_length, ax_fourier, 'b')
    plt.xticks([])
    plt.yticks([])
    plt.show()


    # sample = wy[p1:p3]
    # # sample = ax_new[p1:p3]
    # # sample = af_ax[p1:p3]
    # plt.figure(figsize= (5,4))
    # plt.plot(range(len(sample)), sample, 'b-')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    # plt.subplot(211)
    # # plt.plot(range(len(sample)), sample, 'yo')
    # plt.plot(range(len(sample)), sample, 'b-')
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.subplot(212)
    # plt.plot(range(len(wz)), wz, 'b-')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()


