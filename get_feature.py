import numpy as np
from scipy import interpolate, integrate, stats  # 插值, 积分, 峰度偏度
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from scipy.fftpack import fft, ifft
import get_data
import wave_filter
import setting
import wave_detect


def get_feature_10(wave_ax, wave_ay, wave_az, wave_wx, wave_wy, wave_wz, wave_height, wave_mix):
    ' ax '
    f1 = np.mean(wave_ax)
    f2 = np.std(wave_ax)
    f3 = np.max(wave_ax)
    f4 = np.min(wave_ax)
    f5 = stats.skew(wave_ax)
    f6 = stats.kurtosis(wave_ax)
    ' ay '
    f7 = np.mean(wave_ay)
    f8 = np.std(wave_ay)
    f9 = np.max(wave_ay)
    f10 = np.min(wave_ay)
    f11 = stats.skew(wave_ay)
    f12 = stats.kurtosis(wave_ay)
    ' az '
    f13 = np.mean(wave_az)
    f14 = np.std(wave_az)
    f15 = np.max(wave_az)
    f16 = np.min(wave_az)
    f17 = stats.skew(wave_az)
    f18 = stats.kurtosis(wave_az)
    ' wx '
    f19 = np.mean(wave_wx)
    f20 = np.std(wave_wx)
    f21 = np.max(wave_wx)
    f22 = np.min(wave_wx)
    f23 = stats.skew(wave_wx)
    f24 = stats.kurtosis(wave_wx)
    ' wy '
    f25 = np.mean(wave_wy)
    f26 = np.std(wave_wy)
    f27 = np.max(wave_wy)
    f28 = np.min(wave_wy)
    f29 = stats.skew(wave_wy)
    f30 = stats.kurtosis(wave_wy)
    ' wz '
    f31 = np.mean(wave_wz)
    f32 = np.std(wave_wz)
    f33 = np.max(wave_wz)
    f34 = np.min(wave_wz)
    f35 = stats.skew(wave_wz)
    f36 = stats.kurtosis(wave_wz)
    ' height '
    f37 = np.mean(wave_height)
    f38 = np.std(wave_height)
    f39 = np.max(wave_height)
    f40 = np.min(wave_height)
    f41 = stats.skew(wave_height)
    f42 = stats.kurtosis(wave_height)
    ' mix '
    f43 = np.mean(wave_mix)
    f44 = np.std(wave_mix)
    f45 = np.max(wave_mix)
    f46 = np.min(wave_mix)
    f47 = stats.skew(wave_mix)
    f48 = stats.kurtosis(wave_mix)

    # 添加进 f = []
    f = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                  f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
                  f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
                  f31, f32, f33, f34, f35, f36, f37, f38, f39, f40,
                  f41, f42, f43, f44, f45, f46, f47, f48])
    return f

def get_feature_9(wave_ax, wave_ay, wave_az, wave_wx, wave_wy, wave_wz, wave_mix):
    ' ax '
    f1 = np.mean(wave_ax)
    f2 = np.std(wave_ax)
    f3 = np.max(wave_ax)
    f4 = np.min(wave_ax)
    f5 = stats.skew(wave_ax)
    f6 = stats.kurtosis(wave_ax)
    ' ay '
    f7 = np.mean(wave_ay)
    f8 = np.std(wave_ay)
    f9 = np.max(wave_ay)
    f10 = np.min(wave_ay)
    f11 = stats.skew(wave_ay)
    f12 = stats.kurtosis(wave_ay)
    ' az '
    f13 = np.mean(wave_az)
    f14 = np.std(wave_az)
    f15 = np.max(wave_az)
    f16 = np.min(wave_az)
    f17 = stats.skew(wave_az)
    f18 = stats.kurtosis(wave_az)
    ' wx '
    f19 = np.mean(wave_wx)
    f20 = np.std(wave_wx)
    f21 = np.max(wave_wx)
    f22 = np.min(wave_wx)
    f23 = stats.skew(wave_wx)
    f24 = stats.kurtosis(wave_wx)
    ' wy '
    f25 = np.mean(wave_wy)
    f26 = np.std(wave_wy)
    f27 = np.max(wave_wy)
    f28 = np.min(wave_wy)
    f29 = stats.skew(wave_wy)
    f30 = stats.kurtosis(wave_wy)
    ' wz '
    f31 = np.mean(wave_wz)
    f32 = np.std(wave_wz)
    f33 = np.max(wave_wz)
    f34 = np.min(wave_wz)
    f35 = stats.skew(wave_wz)
    f36 = stats.kurtosis(wave_wz)
    ' mix '
    f37 = np.mean(wave_mix)
    f38 = np.std(wave_mix)
    f39 = np.max(wave_mix)
    f40 = np.min(wave_mix)
    f41 = stats.skew(wave_mix)
    f42 = stats.kurtosis(wave_mix)

    # 添加进 f = []
    f = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                  f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
                  f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
                  f31, f32, f33, f34, f35, f36, f37, f38, f39, f40,
                  f41, f42])
    return f

def fourier_feature(p1, p3, wave_ax, wave_ay, wave_az, wave_wx, wave_wy, wave_wz):
    p_new = np.linspace(p1, p3, 200)

    tck_ax = interpolate.splrep(range(p1, p3), wave_ax, s=0)
    tck_ay = interpolate.splrep(range(p1, p3), wave_ay, s=0)
    tck_az = interpolate.splrep(range(p1, p3), wave_az, s=0)
    tck_wx = interpolate.splrep(range(p1, p3), wave_wx, s=0)
    tck_wy = interpolate.splrep(range(p1, p3), wave_wy, s=0)
    tck_wz = interpolate.splrep(range(p1, p3), wave_wz, s=0)

    ax_new = np.array(interpolate.splev(p_new, tck_ax, der=0))
    ay_new = np.array(interpolate.splev(p_new, tck_ay, der=0))
    az_new = np.array(interpolate.splev(p_new, tck_az, der=0))
    wx_new = np.array(interpolate.splev(p_new, tck_wx, der=0))
    wy_new = np.array(interpolate.splev(p_new, tck_wy, der=0))
    wz_new = np.array(interpolate.splev(p_new, tck_wz, der=0))

    ax_fourier = fourier_transform(ax_new, p_new)
    ay_fourier = fourier_transform(ay_new, p_new)
    az_fourier = fourier_transform(az_new, p_new)
    wx_fourier = fourier_transform(wx_new, p_new)
    wy_fourier = fourier_transform(wy_new, p_new)
    wz_fourier = fourier_transform(wz_new, p_new)

    data_fourier = np.hstack((ax_fourier, ay_fourier))
    data_fourier = np.hstack((data_fourier, az_fourier))
    data_fourier = np.hstack((data_fourier, wx_fourier))
    data_fourier = np.hstack((data_fourier, wy_fourier))
    data_fourier = np.hstack((data_fourier, wz_fourier))

    return data_fourier

def fourier_transform(data, p_new):
    fft_data = fft(data)
    N = len(p_new)
    length = np.arange(N)
    abs_data = np.abs(fft_data)
    # angle_data=np.angle(fft_data)
    normalization_data = abs_data / N  # 归一化处理（双边频谱）
    half_length = length[range(int(N / 2))]  # 取一半区间
    normalization_half_data = normalization_data[range(int(N / 2))]

    # plt.figure()
    # plt.plot(half_length,normalization_half_data,'b')
    # plt.title('单边频谱(归一化)',fontsize=9,color='blue')
    # plt.show()

    return normalization_half_data


# main
def feature_overall(file_9, file_10, data_type, type, file_name):
    start = time.clock()
    # get data
    if data_type == 0:
        times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_data.get_data_10(file_10)
    elif data_type == 1:
        times, ax, ay, az, wx, wy, wz, tx, ty, tz, height = get_data.get_kalman_data_10(file_10)
    rows = len(times)
    thh = 1000  # 波峰波谷

    if data_type == 0:
        # 处理数据 -- kalman filter
        print('需要对10轴传感器数据进行卡尔曼滤波...')
        start_kalman = time.clock()
        ax, ay, az, wx, wy, wz = wave_filter.after_kalman(ax, ay, az, wx, wy, wz, rows)
        data10_after_kalman = setting.save_after_kalman_data10(times, ax, ay, az, wx, wy, wz, tx, ty, tz, height)
        np.savetxt('./data/data_after_kalman/sample10/' + type + '/' + file_name,
               data10_after_kalman, fmt='%s')
        end_kalman = (time.clock() -start_kalman)
        print('滤波完毕, 用时： {}s'.format(end_kalman))


    mix = wave_filter.wave_mix(ax, ay, az, wx, wy, wz) # 平滑滤波
    waves = wave_detect.wave_double(mix, thh)
    # print(len(waves))
    thh = setting.get_thh(mix, waves) # 自适应thh
    # print(thh)
    waves = wave_detect.wave_double(mix, thh)
    # print(len(waves))
    count_10 = setting.calculate_times_count(times)

    feature_10 = []
    for i in range(len(waves)):
        p1 = waves[i][0]
        p2 = waves[i][1]
        p3 = waves[i][2]

        # accelerate
        wave_ax = ax[p1:p3]
        wave_ay = ay[p1:p3]
        wave_az = az[p1:p3]
        # angular velocity
        wave_wx = wx[p1:p3]
        wave_wy = wy[p1:p3]
        wave_wz = wz[p1:p3]
        # height
        wave_height = height[p1:p3]
        # mix
        wave_mix = mix[p1:p3]

        f_normal = get_feature_10(wave_ax, wave_ay, wave_az, wave_wx, wave_wy, wave_wz, wave_height, wave_mix)
        # print('10 f_normal ', len(f_normal))
        f_fourier = fourier_feature(p1, p3, wave_ax, wave_ay, wave_az, wave_wx, wave_wy, wave_wz)
        # print('10 fourier ', len(f_fourier))
        f = np.hstack((f_normal, f_fourier))
        # print('10 f ', len(f))
        feature_10.append(f)
        # print(' ')
    # print('10 feature_10 ', len(feature_10))

    print(' Stage 1 Clear')

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    if data_type == 0:
        times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3 = get_data.get_data_9(file_9)
    elif data_type == 1:
        times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3 = get_data.get_kalman_data_9(file_9)

    rows = len(times)

    if data_type == 0:
        print('需要对9轴传感器数据进行卡尔曼滤波...')
        start_kalman = time.clock()
        ax, ay, az, wx, wy, wz = wave_filter.after_kalman(ax, ay, az, wx, wy, wz, rows)
        data9_after_kalman = setting.save_after_kalman_data9(times, ax, ay, az, wx, wy, wz,
                                                             angle_x, angle_y, angle_z, q0, q1, q2, q3)
        np.savetxt('./data/data_after_kalman/sample9/' + type + '/' + file_name,
                   data9_after_kalman, fmt='%s')
        end_kalman = (time.clock() - start_kalman)
        print('滤波完毕, 用时： {}s'.format(end_kalman))

    mix = wave_filter.wave_mix(ax, ay, az, wx, wy, wz)
    count_9 = setting.calculate_times_count(times)

    feature_9 = []
    for i in range(len(waves)):
        p1 = waves[i][0]
        p2 = waves[i][1]
        p3 = waves[i][2]

        time_p1 = setting.judge_transform(count_9, count_10[p1])
        time_p2 = setting.judge_transform(count_9, count_10[p2])
        time_p3 = setting.judge_transform(count_9, count_10[p3])

        p1 = count_9.index(time_p1)
        p2 = count_9.index(time_p2)
        p3 = count_9.index(time_p3)

        # accelerate
        wave_ax = ax[p1:p3]
        wave_ay = ay[p1:p3]
        wave_az = az[p1:p3]
        # angular velocity
        wave_wx = wx[p1:p3]
        wave_wy = wy[p1:p3]
        wave_wz = wz[p1:p3]
        # mix
        wave_mix = mix[p1:p3]

        f_normal = get_feature_9(wave_ax, wave_ay, wave_az, wave_wx, wave_wy, wave_wz, wave_mix)
        # print('9 f_normal ', len(f_normal))
        f_fourier = fourier_feature(p1, p3, wave_ax, wave_ay, wave_az, wave_wx, wave_wy, wave_wz)
        # print('9 fourier ', len(f_fourier))
        f = np.hstack((f_normal, f_fourier))
        # print('9 f ', len(f))
        feature_9.append(f)
        # print(' ')
    # print('9 feature_9 ', len(feature_10))

    print(' Stage 2 Clear')

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    feature = []
    if len(feature_10) == len(feature_9):
        for i in range(len(feature_9)):
            var_f_9 = feature_9[i]
            var_f_10 = feature_10[i]

            var_f = np.hstack((var_f_10, var_f_9)).reshape(-1, 1)
            scaler = StandardScaler()
            f_s = scaler.fit_transform(var_f).reshape(1, -1)[0]
            feature.append(f_s)
    else:
        print('10轴 与 9轴 特征长度不匹配')


    # np.savetxt("./data/bicycle01.txt", feature, fmt="%f", delimiter=',')

    elapsed = (time.clock() - start)
    print('计算特征值的程序运行总用时： {} s'.format(elapsed))
    # print(len(feature))

    return feature

