import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import interpolate
import train_data_old_version

# 用于插值
def data_interpolate(x, y):
    tck = interpolate.splrep(x, y, s=0)
    xnew1 = np.linspace(x.min(), x.max(), 300)
    ynew1 = interpolate.splev(xnew1, tck, der = 0)
    # 绘图
    plt.figure()
    plt.plot(x, y, 'o', x, y, '--', xnew1, ynew1, '-')
    plt.legend(['data', 'linear', 'cubic'], loc='best')
    plt.show()
    return 0

########################################################################################################################

# 用于拟合的方程
def f_fit(x, A, B, C, D, E, F, G, H, I):  # 用 f_fit函数返回的式子进行拟合
    ''' 8阶函数拟合 '''
    # return A * x**8 + B * x**7 + C * x**6 + D * x**5 + E * x**4 + F * x**3 + G * x**2 + H * x + I

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
    xnew2 = np.linspace(x.min(), x.max(), 300)
    ynew2 = []
    for j in xnew2:
        f_x = f_fit(j, a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8)
        ynew2.append(f_x)

    plt.figure(figsize=(5, 5))

    plt.plot(x, y, 'o', x, y, '--', xnew2, ynew2, '-')
    plt.legend(['data', 'linear', 'curve fit'], loc='best')
    plt.show()
    return 0

########################################################################################################################

if __name__ == '__main__':
    file = './walk_data/heavy_light/heavy/why_heavy_06.txt'

    waves, b, c, d, mix1 = train_data_old_version.walk_wave_2(file)

    for i in range(len(waves)):
        p1 = waves[i][0]
        p2 = waves[i][1]
        p3 = waves[i][2]

        y = mix1[p1:p3]
        x = np.arange(0, len(y))

        temp = data_curvefit(x, y, f_fit)
        plt.savefig('./pic/test/fig_waves_6_' + str(i) + '.png')
        # plt.savefig('./pic/heavy/fig_waves_6_'+str(i)+'.png')
        # plt.close()