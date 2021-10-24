#   用于读取数据文件中的数据
#   get_data_9 和 get_data_10 分别用于读取 9轴 和 10轴传感器的数据

import pandas as pd

'输入 file        输出 times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3'
def get_data_9(file):
    data = pd.read_csv(file, header=None)[2:].values
    rows = data.shape[0]
    num = 100

    times = []
    ax, ay, az = [], [], []
    wx, wy, wz = [], [], []
    angle_x, angle_y, angle_z = [], [], []
    q0, q1, q2, q3 = [], [], [], []

    # 筛选x轴,y轴,z轴数据
    for i in range(rows - 1):
        sample = data[i]
        if len(sample[0].split()) == 15:
            # time
            t_time = sample[0].split()[1]
            # accelerate
            t_ax = float(sample[0].split()[2]) * num * 9.80665
            t_ay = float(sample[0].split()[3]) * num * 9.80665
            t_az = float(sample[0].split()[4]) * num * 9.80665
            # angular velocity
            t_wx = float(sample[0].split()[5]) * num
            t_wy = float(sample[0].split()[6]) * num
            t_wz = float(sample[0].split()[7]) * num
            # Euler angle
            t_angle_x = float(sample[0].split()[8])
            t_angle_y = float(sample[0].split()[9])
            t_angle_z = float(sample[0].split()[10])
            # quaternion
            t_q0 = float(sample[0].split()[11])
            t_q1 = float(sample[0].split()[12])
            t_q2 = float(sample[0].split()[13])
            t_q3 = float(sample[0].split()[14])

            times.append(t_time)
            ax.append(t_ax); ay.append(t_ay); az.append(t_az)
            wx.append(t_wx); wy.append(t_wy); wz.append(t_wz)
            angle_x.append(t_angle_x); angle_y.append(t_angle_y); angle_z.append(t_angle_z)
            q0.append(t_q0); q1.append(t_q1); q2.append(t_q2); q3.append(t_q3)

    return times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3


'输入 file        输出 times, ax, ay, az, wx, wy, wz, tx, ty, tz, height'
def get_data_10(file):
    data = pd.read_csv(file, header=None)[2:].values
    rows = data.shape[0]
    num = 100

    times = []
    ax, ay, az = [], [], []
    wx, wy, wz = [], [], []
    tx, ty, tz = [], [], []
    height = []

    # 筛选x轴,y轴,z轴数据
    for i in range(rows - 1):
        sample = data[i]
        # time
        if len(sample[0].split()) == 13:
            t_time = sample[0].split()[1]
            # accelerate
            t_ax = float(sample[0].split()[2]) * num * 9.80665
            t_ay = float(sample[0].split()[3]) * num * 9.80665
            t_az = float(sample[0].split()[4]) * num * 9.80665
            # angular velocity
            t_wx = float(sample[0].split()[5]) * num
            t_wy = float(sample[0].split()[6]) * num
            t_wz = float(sample[0].split()[7]) * num
            # tesla
            t_tx = float(sample[0].split()[8]) * num
            t_ty = float(sample[0].split()[9]) * num
            t_tz = float(sample[0].split()[10]) * num
            # height
            t_height = float(sample[0].split()[12]) * num

            times.append(t_time)
            ax.append(t_ax); ay.append(t_ay); az.append(t_az)
            wx.append(t_wx); wy.append(t_wy); wz.append(t_wz)
            tx.append(t_tx); ty.append(t_ty); tz.append(t_tz)
            height.append(t_height)

    return times, ax, ay, az, wx, wy, wz, tx, ty, tz, height


def get_kalman_data_10(file):
    data = pd.read_csv(file, header=None)[2:].values
    rows = data.shape[0]
    num = 100

    times = []
    ax, ay, az = [], [], []
    wx, wy, wz = [], [], []
    tx, ty, tz = [], [], []
    height = []

    for i in range(rows - 1):
        sample = data[i]

        # time
        t_time = str(sample[0].split(' ')[0])
        # accelerate
        t_ax = float(sample[0].split(' ')[1]) * num * 9.80665
        t_ay = float(sample[0].split(' ')[2]) * num * 9.80665
        t_az = float(sample[0].split(' ')[3]) * num * 9.80665
        # angular velocity
        t_wx = float(sample[0].split(' ')[4]) * num
        t_wy = float(sample[0].split(' ')[5]) * num
        t_wz = float(sample[0].split(' ')[6]) * num
        # tesla
        t_tx = float(sample[0].split(' ')[7]) * num
        t_ty = float(sample[0].split(' ')[8]) * num
        t_tz = float(sample[0].split(' ')[9]) * num
        # height
        t_height = float(sample[0].split(' ')[10]) * num
        
        times.append(t_time)
        ax.append(t_ax); ay.append(t_ay); az.append(t_az)
        wx.append(t_wx); wy.append(t_wy); wz.append(t_wz)
        tx.append(t_tx); ty.append(t_ty); tz.append(t_tz)
        height.append(t_height)

    return times, ax, ay, az, wx, wy, wz, tx, ty, tz, height

def get_kalman_data_9(file):
    data = pd.read_csv(file, header=None)[2:].values
    rows = data.shape[0]
    num = 100

    times = []
    ax, ay, az = [], [], []
    wx, wy, wz = [], [], []
    angle_x, angle_y, angle_z = [], [], []
    q0, q1, q2, q3 = [], [], [], []

    # 筛选x轴,y轴,z轴数据
    for i in range(rows - 1):
        sample = data[i]

        # time
        t_time = sample[0].split()[0]
        # accelerate
        t_ax = float(sample[0].split()[1]) * num * 9.80665
        t_ay = float(sample[0].split()[2]) * num * 9.80665
        t_az = float(sample[0].split()[3]) * num * 9.80665
        # angular velocity
        t_wx = float(sample[0].split()[4]) * num
        t_wy = float(sample[0].split()[5]) * num
        t_wz = float(sample[0].split()[6]) * num
        # Euler angle
        t_angle_x = float(sample[0].split()[7])
        t_angle_y = float(sample[0].split()[8])
        t_angle_z = float(sample[0].split()[9])
        # quaternion
        t_q0 = float(sample[0].split()[10])
        t_q1 = float(sample[0].split()[11])
        t_q2 = float(sample[0].split()[12])
        t_q3 = float(sample[0].split()[13])
        
        times.append(t_time)
        ax.append(t_ax); ay.append(t_ay); az.append(t_az)
        wx.append(t_wx); wy.append(t_wy); wz.append(t_wz)
        angle_x.append(t_angle_x); angle_y.append(t_angle_y); angle_z.append(t_angle_z)
        q0.append(t_q0); q1.append(t_q1); q2.append(t_q2); q3.append(t_q3)

    return times, ax, ay, az, wx, wy, wz, angle_x, angle_y, angle_z, q0, q1, q2, q3
