from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import logging
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':

    '''  步态识别  '''
    # # get test data
    # test = pd.read_csv('./walk_data/person_identify/test.txt', header=None).values
    # # get train data
    # data_why = pd.read_csv('./walk_data/person_identify/train_why_data.txt', header=None).values
    # data_wgx = pd.read_csv('./walk_data/person_identify/train_wgx_data.txt', header=None).values
    # data_whf = pd.read_csv('./walk_data/person_identify/train_whf_data.txt', header=None).values
    # data_gwt = pd.read_csv('./walk_data/person_identify/train_gwt_data.txt', header=None).values
    # # get train label
    # label_why = np.ones([1, len(data_why)])[0]  # 1 == why
    # label_wgx = np.ones([1, len(data_wgx)])[0] * 2  # 2 == wgx
    # label_whf = np.ones([1, len(data_whf)])[0] * 3  # 3 == whf
    # label_gwt = np.ones([1, len(data_gwt)])[0] * 4  # 4 == gwt
    # # conbine
    # label = np.hstack((label_why, label_wgx, label_whf, label_gwt))
    # train = np.vstack((data_why, data_wgx, data_whf, data_gwt))

########################################################################################################################

    '''  负重识别  '''
    # get test data
    test = pd.read_csv('./walk_data/heavy_light/feature_test.txt', header=None).values
    # get train data
    data_heavy = pd.read_csv('./walk_data/heavy_light/feature_train_heavy.txt', header=None).values
    data_light = pd.read_csv('./walk_data/heavy_light/feature_train_light.txt', header=None).values
    # get train label
    label_heavy = np.ones([1, len(data_heavy)])[0] * 10000  # 10000 == heavy
    label_light = np.ones([1, len(data_light)])[0] * (-10000)  # -10000 == light
    # conbine
    label = np.hstack((label_heavy, label_light))
    train = np.vstack((data_heavy, data_light))

########################################################################################################################

    # 归一化数据并作图
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)


    # 启动
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    # 搭建模型
    # l1
    l1 = tf.keras.layers.Dense(units=1, input_shape=[30])
    model1 = tf.keras.Sequential([l1])
    model1.compile(loss='mean_squared_error',
                   optimizer=tf.keras.optimizers.Adam(0.1))
    history1 = model1.fit(train, label, epochs=500, verbose=False)
    print("Finished training the model 1")

    # 绘图
    plt.xlabel('Epoch Number')
    plt.ylabel("Loss Magnitude")
    plt.plot(history1.history['loss'])

    # 模型预测
    scaler = StandardScaler()
    scaler.fit(test)
    test = scaler.transform(test)
    var = model1.predict([test])

    '''  heavy-light  '''
    light, heavy = [], []
    # 输出结果
    for j in range(len(var)):
        print(var[j])
        if var[j] <= 0:
            light.append(var[j])
        elif var[j] > 0:
            heavy.append(var[j])

    print('accuracy: {}'.format(np.mean(var)))
    print('light: {}    heavy: {}'.format(len(light), len(heavy)))

    # plt.plot(range(len(var)), np.zeros([1, len(var)])[0], '-', range(len(var)), var, 'o')
    # plt.show()


# test = pd.read_csv('./walk_data/heavy_light/test.txt', header=None).values
# scaler = StandardScaler()
# test = scaler.transform(test)
# var = model1.predict([test])
# light, heavy = [], []
# for j in range(len(var)):
#     if var[j] <= 0:
#         light.append(var[j])
#     elif var[j] > 0:
#         heavy.append(var[j])
# print('accuracy: {}'.format(np.mean(var)))
# print('light: {}    heavy: {}'.format(len(light), len(heavy)))


# plt.plot(range(len(var)), np.zeros([1, len(var)])[0], '-', range(len(var)), var, 'o')
# plt.show()