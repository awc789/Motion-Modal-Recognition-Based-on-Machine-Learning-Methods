# basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# personal
import setting
from get_data import get_kalman_data_9, get_kalman_data_10
import wave_filter
import wave_detect

# public
import glob
from scipy import interpolate, integrate, stats  # 插值, 积分, 峰度偏度
from sympy import diff, symbols  # 求导, 符号
from scipy import optimize
from sklearn.preprocessing import StandardScaler
import time


# 用于拟合的方程
def f_fit(x, A, B, C, D, E, F, G, H, I):  # 用 f_fit函数返回的式子进行拟合
    ''' Legendre 多项式拟合 '''
    x0 = 1
    x1 = x
    x2 = (1/2) * (3 * x**2 - 1)
    x3 = (1/2) * (5 * x**3 - 3 * x)
    x4 = (1/8) * (35 * x**4 - 30 * x**2 + 3)
    x5 = (1/8) * (63 * x**5 - 70 * x**3 + 15 * x)
    x6 = (1/16) * (231 * x**6 - 315 * x**4 + 105 * x**2 - 5)
    x7 = (1/16) * (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x)
    x8 = (1/128) * (6435 * x**8 - 12012 * x**6 + 6930 * x**4 - 1260 * x**2 + 35)
    return ((A * x8) + (B * x7) + (C * x6) + (D * x5) + (E * x4) + (F * x3) + (G * x2) + (H * x1) + (I * x0))

# 曲线拟合
def data_curvefit(x, y, f_fit):
    p_fit, prov = optimize.curve_fit(f_fit, x, y)  # p_fit 拟合系数
    # 读取拟合系数
    for i in range(len(p_fit)):
        a_0 = p_fit[0]
        a_1 = p_fit[1]
        a_2 = p_fit[2]
        a_3 = p_fit[3]
        a_4 = p_fit[4]
        a_5 = p_fit[5]
        a_6 = p_fit[6]
        a_7 = p_fit[7]
        a_8 = p_fit[8]
    # get function
    t = symbols('t', real=True)
    f_x = f_fit(t, a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8)
    return f_x

def cnn_data(var, p1, p3):
    # print(var, '  and  ', p3-p1)
    p_new = np.linspace(p1, p3, 300)  # 返回p1-p3之间均匀间隔的数字
    tck_var = interpolate.splrep(range(len(var)), var, s=0)  # 样条插值
    var_new = np.array(interpolate.splev(p_new, tck_var, der=0)) # 生成新的等成长度的数据

    # 获得导数
    data_var = var_new
    # curve fit
    t = symbols('t', real=True)
    f_x = data_curvefit(np.arange(0, len(data_var)), data_var, f_fit)
    # diff
    diff_1, diff_2 = [], []
    for i in range(len(data_var)):
        # 1 - diff
        var_x1 = diff(f_x, t, 1).subs(t, data_var[i])
        diff_1.append(var_x1)
        # 2 - diff
        var_x2 = diff(f_x, t, 2).subs(t, data_var[i])
        diff_2.append(var_x2)

    # 获得差分
    diff_var = pd.Series(var_new)
    liner_diff_1 = diff_var.diff()
    liner_diff_2 = liner_diff_1.diff()

    liner_diff_1 = list(liner_diff_1)
    liner_diff_2 = list(liner_diff_2)
    liner_diff_1[0], liner_diff_2[0], liner_diff_2[1] = 0, 0, 0

    return var_new, diff_1, diff_2, liner_diff_1, liner_diff_2

if __name__ == '__main__':
    start = time.clock()
    file_list = setting.get_type()
    path_10 = './data/sample10/'
    path_9 = './data/sample9/'

    path_after_kalman = './data/data_after_kalman/'

    for type in file_list:
        number = 1
        files = glob.glob(path_10 + type + '/*.txt')

        for file in files:
            file_name = file.split('_')[-1]
            # 获得两个对应的数据文件
            file_10 = path_after_kalman + 'sample10/' + type + '/' + file_name
            file_9 = path_after_kalman + 'sample9/' + type + '/' + file_name

            # start program
            print('************************************************************************************************')
            print('type: {} , file name: {}'.format(type, file_name))

            # get data
            times_10, ax_10, ay_10, az_10, wx_10, wy_10, wz_10, tx_10, ty_10, tz_10, height_10 = get_kalman_data_10(file_10)
            times_9, ax_9, ay_9, az_9, wx_9, wy_9, wz_9, angle_x_9, angle_y_9, angle_z_9, q0_9, q1_9, q2_9, q3_9 = get_kalman_data_9(file_9)

            # 获得数据的 peak-valley
            thh = 1000  # 波峰波谷
            mix = wave_filter.wave_mix(ax_10, ay_10, az_10, wx_10, wy_10, wz_10)  # 平滑滤波
            waves = wave_detect.wave_double(mix, thh)
            # print(len(waves))
            thh = setting.get_thh(mix, waves)  # 自适应thh
            # print(thh)
            waves = wave_detect.wave_double(mix, thh)
            # print(len(waves))
            count_10 = setting.calculate_times_count(times_10)


            for i in range(len(waves)):
                p1 = waves[i][0]
                p2 = waves[i][1]
                p3 = waves[i][2]

                # # 10 轴数据
                # ax_wave_10 = ax_10[p1:p3] # a
                # ay_wave_10 = ay_10[p1:p3]
                # az_wave_10 = az_10[p1:p3]
                # wx_wave_10 = wx_10[p1:p3] # w
                # wy_wave_10 = wy_10[p1:p3]
                # wz_wave_10 = wz_10[p1:p3]
                # tx_wave_10 = tx_10[p1:p3] # t
                # ty_wave_10 = ty_10[p1:p3]
                # tz_wave_10 = tz_10[p1:p3]
                # height_wave_10 = height_10[p1:p3] # height
                # # 9 轴数据
                # ax_wave_9 = ax_9[p1:p3]  # a
                # ay_wave_9 = ay_9[p1:p3]
                # az_wave_9 = az_9[p1:p3]
                # wx_wave_9 = wx_9[p1:p3]  # w
                # wy_wave_9 = wy_9[p1:p3]
                # wz_wave_9 = wz_9[p1:p3]
                # angle_x_wave_9 = angle_x_9[p1:p3] # angle
                # angle_y_wave_9 = angle_y_9[p1:p3]
                # angle_z_wave_9 = angle_z_9[p1:p3]
                # q0_wave_9 = q0_9[p1:p3] # q
                # q1_wave_9 = q1_9[p1:p3]
                # q2_wave_9 = q2_9[p1:p3]
                # q3_wave_9 = q3_9[p1:p3]

                CNN_PIC = []
                # 10 轴数据
                CNN_PIC.append(ax_10[p1:p3])  # a
                CNN_PIC.append(ax_9[p1:p3])
                CNN_PIC.append(ay_10[p1:p3])
                CNN_PIC.append(ay_9[p1:p3])
                CNN_PIC.append(az_10[p1:p3])
                CNN_PIC.append(az_9[p1:p3])

                CNN_PIC.append(wx_10[p1:p3])  # w
                CNN_PIC.append(wx_9[p1:p3])
                CNN_PIC.append(wy_10[p1:p3])
                CNN_PIC.append(wy_9[p1:p3])
                CNN_PIC.append(wz_10[p1:p3])
                CNN_PIC.append(wz_9[p1:p3])

                CNN_PIC.append(angle_x_9[p1:p3])  # angle
                CNN_PIC.append(angle_y_9[p1:p3])
                CNN_PIC.append(angle_z_9[p1:p3])

                CNN_PIC.append(q0_9[p1:p3])  # q
                CNN_PIC.append(q1_9[p1:p3])
                CNN_PIC.append(q2_9[p1:p3])
                CNN_PIC.append(q3_9[p1:p3])

                CNN_PIC.append(height_10[p1:p3])  # height

                CNN_PIC_FINAL = []
                DIFF_1, DIFF_2 = [], []
                LINER_DIFF_1, LINER_DIFF_2 = [], []

                if [] in CNN_PIC: # 防止报错
                    pass
                else:
                    for temp in CNN_PIC:
                        var_new, diff_1, diff_2, liner_diff_1, liner_diff_2 = cnn_data(temp, p1, p3)
                        CNN_PIC_FINAL.append(np.array(var_new))
                        DIFF_1.append(np.array(diff_1))
                        DIFF_2.append(np.array(diff_2))
                        LINER_DIFF_1.append(np.array(liner_diff_1))
                        LINER_DIFF_2.append(np.array(liner_diff_1))

                    result = CNN_PIC_FINAL
                    for temp in DIFF_1[:6]:
                        result.append(temp)
                    for temp in DIFF_2[:6]:
                        result.append(temp)
                    for temp in LINER_DIFF_1[:6]:
                        result.append(temp)
                    for temp in LINER_DIFF_2[:6]:
                        result.append(temp)

                    np.savetxt('./data/data_cnn/' + type + '/' + type + str(number) + '.txt',
                           result, fmt="%f", delimiter=',')

                    number += 1
                    if number % 10 == 0:
                        print('Now: No.', number)
        print('************************************************************************************************')

    elapsed = (time.clock() - start)
    hour = int(elapsed // 3600)
    minute = int((elapsed % 3600) // 60)
    second = int(elapsed - 3600 * hour - 60 * minute)
    print(' ')
    print(' ')
    print(' ')
    print("Time used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))


