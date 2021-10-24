#   一些额外的操作
#   alculate_times_count 用于将传感器时间数据转换成计数数据，方便后期的10轴与9轴的对接
#   judge_transform 用于对接9轴和10轴数据的时间节点
#   get_type 用于获取运动数据的类型
#   get_thh 用于实现 peak valley 中波峰波谷查找时候最小间隔的 thh 自适应
#   save_after_kalman_data10 和 save_after_kalman_data9 用于存储卡尔曼滤波后的数据，方便保存

import numpy as np
import glob

'输入 times        输出 count'
def calculate_times_count(times):
    rows = len(times)
    b_time = times[0]
    detail = b_time.split(':')
    b_hour = int(detail[0])
    b_minute = int(detail[1])
    second = detail[2]
    second = second.split('.')
    b_second, b_min_second = int(second[0]), int(second[1])
    count = []

    b_number = (b_hour * 3600 + b_minute * 60 + b_second) * 200 + int(b_min_second / 5)

    for i in range(rows):
        time = times[i]
        detail = time.split(':')
        hour = int(detail[0])
        minute = int(detail[1])
        second = detail[2]
        second = second.split('.')
        second, min_second = int(second[0]), int(second[1])

        number = (hour * 3600 + minute * 60 + second) * 200 + int(min_second / 5)
        var_count = number - b_number
        count.append(var_count)

    return count


'10_axis and 9_axis 转换'
def judge_transform(list, number):
    if number in list:
        return number
    else:
        new_number = number
        while new_number not in list:
            new_number += 1
        return new_number


'获得运动的类型名称'
def get_type():
    path = './data/sample10/'
    files = glob.glob(path + '*')

    list = []
    for file in files:
        names = file.split('/')
        list.append(names[-1])

    return list


'输入 waves        输出 thh'
def get_thh(mix, waves):
    rows = len(waves)
    top, bottom = [], []
    for i in range(rows):
        p1 = waves[i][0]
        p2 = waves[i][1]
        p3 = waves[i][2]

        valley = mix[p1]
        peak = mix[p2]
        top.append(peak)
        bottom.append(valley)

    # data = mix[waves[0][0]:waves[rows-1][2]]
    # mean = np.mean(data)

    data_max = np.mean(top)
    data_min = np.mean(bottom)
    height = data_max - data_min
    thh = height / 3

    return thh


'save after kalman data'
def save_after_kalman_data10(times, ax, ay, az, wx, wy, wz, tx, ty, tz, height):
    rows = len(times)
    data10 = []
    for i in range(rows):
        data = np.array([times[i], ax[i], ay[i], az[i], wx[i], wy[i], wz[i], tx[i], ty[i], tz[i], height[i]])
        data10.append(data)

    data10 = np.array(data10)
    return data10

def save_after_kalman_data9(times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3):
    rows = len(times)
    data9 = []
    for i in range(rows):
        data = np.array([times[i], ax[i], ay[i], az[i], wx[i], wy[i], wz[i], angle_x[i], angle_y[i], angle_z[i],
                        q0[i], q1[i], q2[i], q3[i]])
        data9.append(data)

    data9 = np.array(data9)
    return data9

