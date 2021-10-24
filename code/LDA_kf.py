from __future__ import absolute_import, division, print_function, unicode_literals
from KalmanFilter import kalman_filter_1, get_data, walk_wave
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import glob
import logging
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import time

# light  label = -1
def light_data(file, feature, label):
    measurements = np.array(get_data(file))
    x_k, y_k, z_k = kalman_filter_1(measurements)  # data after kalman filter
    waves = walk_wave(x_k, y_k, z_k)

    for i in range(len(waves)):
        p1 = waves[i][0]
        p2 = waves[i][1]
        p3 = waves[i][2]

        x = x_k[p1:p3]
        y = y_k[p1:p3]
        z = z_k[p1:p3]

        p_new = np.linspace(p1, p3, 100)

        tck_x = interpolate.splrep(range(p1, p3), x, s=0)
        tck_y = interpolate.splrep(range(p1, p3), y, s=0)
        tck_z = interpolate.splrep(range(p1, p3), z, s=0)

        x_new = np.array(interpolate.splev(p_new, tck_x, der=0))
        y_new = np.array(interpolate.splev(p_new, tck_y, der=0))
        z_new = np.array(interpolate.splev(p_new, tck_z, der=0))
        temp = np.array([x_new, y_new, z_new]).flatten()
        feature.append(temp)
        label.append(-1)

    return feature, label
# heavy  label = 1
def heavy_data(file, feature, label):
    measurements = np.array(get_data(file))
    x_k, y_k, z_k = kalman_filter_1(measurements)  # data after kalman filter
    waves = walk_wave(x_k, y_k, z_k)

    for i in range(len(waves)):
        p1 = waves[i][0]
        p2 = waves[i][1]
        p3 = waves[i][2]

        x = x_k[p1:p3]
        y = y_k[p1:p3]
        z = z_k[p1:p3]

        p_new = np.linspace(p1, p3, 100)

        tck_x = interpolate.splrep(range(p1, p3), x, s=0)
        tck_y = interpolate.splrep(range(p1, p3), y, s=0)
        tck_z = interpolate.splrep(range(p1, p3), z, s=0)

        x_new = np.array(interpolate.splev(p_new, tck_x, der=0))
        y_new = np.array(interpolate.splev(p_new, tck_y, der=0))
        z_new = np.array(interpolate.splev(p_new, tck_z, der=0))
        temp = np.array([x_new, y_new, z_new]).flatten()
        feature.append(temp)
        label.append(1)

    return feature, label
# test
def test_data(file, test, test_label, n):
    measurements = np.array(get_data(file))
    x_k, y_k, z_k = kalman_filter_1(measurements)  # data after kalman filter
    waves = walk_wave(x_k, y_k, z_k)

    for i in range(len(waves)):
        p1 = waves[i][0]
        p2 = waves[i][1]
        p3 = waves[i][2]

        x = x_k[p1:p3]
        y = y_k[p1:p3]
        z = z_k[p1:p3]

        p_new = np.linspace(p1, p3, 100)

        tck_x = interpolate.splrep(range(p1, p3), x, s=0)
        tck_y = interpolate.splrep(range(p1, p3), y, s=0)
        tck_z = interpolate.splrep(range(p1, p3), z, s=0)

        x_new = np.array(interpolate.splev(p_new, tck_x, der=0))
        y_new = np.array(interpolate.splev(p_new, tck_y, der=0))
        z_new = np.array(interpolate.splev(p_new, tck_z, der=0))
        temp = np.array([x_new, y_new, z_new]).flatten()
        test.append(temp)
        test_label.append(n)

    return test, test_label

if __name__ == '__main__':
    ''' time '''
    start = time.clock()

    # single
    # file_light = './walk_data/heavy_light/light/why_light_01.txt'
    # file_light = './walk_data/heavy_light/test/test_light_01.txt'
    # file_heavy = './walk_data/heavy_light/heavy/why_heavy_01.txt'
    # file_heavy = './walk_data/heavy_light/test/test_heavy_01.txt'

    # multiple
    path_heavy = './walk_data/heavy_light/heavy/'
    files_heavy = glob.glob(path_heavy + '*.txt')

    path_light = './walk_data/heavy_light/light/'
    files_light = glob.glob(path_light + '*.txt')

########################################################################################################################

    feature = []
    label = []

    for file_light in files_light:
        ''' light '''
        feature, label = light_data(file_light, feature, label)  # -1 == light
    for file_heavy in files_heavy:
        ''' heavy '''
        feature, label = heavy_data(file_heavy, feature, label)  # 1 == heavy

    lda = LinearDiscriminantAnalysis(n_components=1)
    # data_lda = lda.fit(np.array(feature), np.array(label)).transform(np.array(feature))
    data_lda = lda.fit_transform(np.array(feature), np.array(label))

    plt.figure(1)
    plt.plot(range(len(label)), np.ones([1, len(label)])[0] * 0, 'b-')
    plt.plot(range(len(data_lda)), data_lda, 'ro')
    plt.plot(range(len(label)), label, 'g--')
    plt.show()

########################################################################################################################

    test = []
    test_label = []

    file_test_heavy = './walk_data/heavy_light/test/test_heavy_03.txt'
    file_test_light = './walk_data/heavy_light/test/test_light_01.txt'

    test, test_label = test_data(file_test_heavy, test, test_label, 1)  # 1 == heavy
    test, test_label = test_data(file_test_light, test, test_label, -1)  # -1 == light
    test_lda = lda.transform(test)

    acc = 0
    for i in range(len(test_lda)):
        if test_lda[i] < 0:
            temp = -1
        elif test_lda[i] > 0:
            temp = 1
        elif test_lda == 0:
            temp = 0
        if temp == test_label[i]:
            acc += 1
    print('accuracy: %.4f' % (acc/len(test_lda)))

    plt.figure(2)
    plt.plot(range(len(test)), np.ones([1, len(test)])[0] * 0, 'b-')
    plt.plot(range(len(test_lda)), test_lda, 'ro')
    plt.plot(range(len(test_label)), test_label, 'g--')
    plt.show()

    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

''' from sklearn.externals import joblib '''
''' joblib.dump(lda, 'LDA.pkl') '''
''' lda = joblib.load('LDA.pkl') '''

########################################################################################################################

    # # Using TensorFlow to test
    # test = []
    # file_test = './walk_data/heavy_light/test/test_light_01.txt'
    # test = test_data(file_test, test)
    #
    #
    # # 归一化数据并作图
    # scaler = StandardScaler()
    # scaler.fit(feature)
    # feature = scaler.transform(feature)
    #
    #
    # # 启动
    # logger = tf.get_logger()
    # logger.setLevel(logging.ERROR)
    #
    # # 搭建模型
    # # l1
    # l1 = tf.keras.layers.Dense(units=1, input_shape=[300])
    # model1 = tf.keras.Sequential([l1])
    # model1.compile(loss='mean_squared_error',
    #                optimizer=tf.keras.optimizers.Adam(0.1))
    # history1 = model1.fit(feature, data_lda, epochs=500, verbose=False)
    # print("Finished training the model 1")
    #
    # # 绘图
    # plt.xlabel('Epoch Number')
    # plt.ylabel("Loss Magnitude")
    # plt.plot(history1.history['loss'])
    #
    # # 模型预测
    # scaler = StandardScaler()
    # scaler.fit(test)
    # test = scaler.transform(test)
    # var = model1.predict([test])
    #
    # '''  heavy-light  '''
    # light, heavy = [], []
    # temp = []
    # # 输出结果
    # for j in range(len(var)):
    #     print(var[j])
    #     if var[j] <= 0:
    #         light.append(var[j])
    #         temp.append(-1)
    #     elif var[j] > 0:
    #         heavy.append(var[j])
    #         temp.append(1)
    #
    # print('accuracy: {}'.format(np.mean(temp)))
    # print('light: {}    heavy: {}'.format(len(light), len(heavy)))
    #
    # plt.plot(range(len(var)), np.zeros([1, len(var)])[0], '-', range(len(var)), var, 'o')
    # plt.show()

########################################################################################################################

