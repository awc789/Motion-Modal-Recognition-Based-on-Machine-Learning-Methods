#   用于进行数据的滤波
#   kalman_filter_1 用于对传感器采集的数据进行滤波，排除异常点
#   after_kalman 通过调用 kalman_filter_1 完成对数据点的卡尔曼滤波
#   wave_mix 则用于对进行寻找 peak valley 的时候平滑曲线，方便滤波

from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import time

'kalman filter'
def kalman_filter_1(measurements):
    # 初始值
    initial_state_mean = [measurements[0, 0],
                          0,
                          measurements[0, 1],
                          0,
                          measurements[0, 2],
                          0]
    # 转换矩阵
    transition_matrix = [[1, 1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 1]]
    # 观测矩阵
    observation_matrix = [[1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0]]

    kf1 = KalmanFilter(transition_matrices=transition_matrix,
                       observation_matrices=observation_matrix,
                       initial_state_mean=initial_state_mean)

    kf1 = kf1.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

    # plt.figure(1)
    # times = range(measurements.shape[0])
    # plt.plot(times, measurements[:, 0], 'bo', times, smoothed_state_means[:, 0], 'b--', )
    # plt.plot(times, measurements[:, 1], 'ro', times, smoothed_state_means[:, 2], 'r--', )
    # plt.plot(times, measurements[:, 2], 'go', times, smoothed_state_means[:, 4], 'g--', )
    # plt.show()

    return smoothed_state_means[:, 0], smoothed_state_means[:, 2], smoothed_state_means[:, 4]


'输入 ax, ay, az, wx, wy, wz, rows        输出 滤波后分的 ax, ay, az, wx, wy, wz'
def after_kalman(ax, ay, az, wx, wy, wz, rows):
    # kalman filter
    temp_a = []
    for i in range(rows):
        var1 = ax[i]
        var2 = ay[i]
        var3 = az[i]
        temp_a.append([var1, var2, var3])

    print('加速度滤波...')
    time_start = time.clock()
    measurements_a = np.array(temp_a)
    ax, ay, az = kalman_filter_1(measurements_a)  # data after kalman filter
    time_end = time.clock()
    print('加速度滤波完成, 用时 {} s'.format(time_end - time_start))

    temp_w = []
    for i in range(rows):
        var1 = wx[i]
        var2 = wy[i]
        var3 = wz[i]
        temp_w.append([var1, var2, var3])

    print('陀螺仪滤波...')
    time_start = time.clock()
    measurements_w = np.array(temp_w)
    wx, wy, wz = kalman_filter_1(measurements_w)  # data after kalman filter
    time_end = time.clock()
    print('陀螺仪滤波完成, 用时 {} s'.format(time_end - time_start))

    return ax, ay, az, wx, wy, wz


'wave mix'
def wave_mix(ax, ay, az, wx, wy, wz):
    x, y, z = wx, wy, wz
    rows = len(ax)

    # 79 and 59
    num1 = 79
    num2 = 59
    b = x[0:num1]
    c = y[0:num1]

    mix_w1, mix_w2 = [], []
    for i in range(num1):
        var = c[i] - b[i]
        mix_w1.append(var)
    for i in range(num2):
        var = c[i] - b[i]
        mix_w2.append(var)

    # 一次滤波
    for i in range(num1, rows):  # 9 --- rows-1
        sum_1 = 0
        sum_2 = 0
        for j in range(num1 + 1):
            sum_1 = sum_1 + x[i + j - num1]  # 每10个求和
            sum_2 = sum_2 + y[i + j - num1]
        avg_1 = sum_1 / (num1 + 1)
        avg_2 = sum_2 / (num1 + 1)
        mix_w1.append((avg_2 - avg_1))  # 把 y轴的平均值 和 x轴的平均值 相减放入mix
        # b.append(avg_1)  # 把 x轴的均值 存入b
        # c.append(avg_2)  # 把 y轴的均值 存入c

    # 二次滤波
    for i in range(num2, rows):
        sum_m = 0
        for j in range(num2 + 1):
            sum_m = sum_m + mix_w1[i + j - num2]  # 每6个求和
        avg_m = sum_m / (num2 + 1)
        mix_w2.append(avg_m)

    ''' accelerate '''
    x, y, z = ax, ay, az
    mix_a1 = []
    mix_a2 = []
    for i in range(rows):
        var_x = x[i]
        var_y = y[i]
        var_z = z[i]

        var_a1 = np.sqrt(var_x * var_x + var_y * var_y + var_z * var_z)
        mix_a1.append(var_a1)

        var_a2 = var_z - var_x + var_y
        mix_a2.append(var_a2)

    mix = []
    for i in range(len(mix_w2)):
        var1 = mix_w2[i]
        var2 = mix_a1[i]
        var3 = mix_a2[i]

        var_f = var1 + var2 + var3
        mix.append(var_f)

    num3 = 49

    mix_final = mix[0:num3]
    for i in range(num3, rows):
        sum = 0
        for j in range(num3 + 1):
            sum = sum + mix[i + j - num3]
        avg = sum / (num3 + 1)

        mix_final.append(avg)

    return mix_final