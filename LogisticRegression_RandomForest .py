from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # standerlize
from sklearn.ensemble import RandomForestClassifier
import train_data_old_version
import glob

# logistic regreession        输入train, label        输出prediction, w, b
def logistic_regression(train, label, test):
    lr = LogisticRegression()
    lr.fit(train, label)
    print("LR-score:", lr.score(train, label))

    w = lr.coef_[0]
    b = lr.intercept_
    prediction = lr.predict(test)

    return prediction, w, b

# random forest        输入train, label        输出prediction
def random_forest(train, label, test):
    rf = RandomForestClassifier()
    rf.fit(train, label)
    print("RF-score:", rf.score(train, label))

    prediction = rf.predict(test)

    return prediction


if __name__ == '__main__':

    '''  步态识别  '''
    # # 标准化
    # scaler = StandardScaler()
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
    # # get test data
    # test = pd.read_csv('./walk_data/heavy_light/feature_test_heavy.txt', header=None).values
    # # get train data
    # data_heavy = pd.read_csv('./walk_data/heavy_light/feature_train_heavy.txt', header=None).values
    # data_light = pd.read_csv('./walk_data/heavy_light/feature_train_light.txt', header=None).values
    # # get train label
    # label_heavy = np.ones([1, len(data_heavy)])[0] * 1  # 10000 == heavy
    # label_light = np.ones([1, len(data_light)])[0] * (0)  # -10000 == light
    # # conbine
    # label = np.hstack((label_heavy, label_light))
    # train = np.vstack((data_heavy, data_light))

########################################################################################################################

    '''  毕业设计  '''

    test = pd.read_csv('./data/select_data/bicycle/bicycle01.txt', header=None).values
    test = pd.read_csv('./data/select_data/downstairs/downstairs01.txt', header=None).values
    test = pd.read_csv('./data/select_data/heavy/heavy01.txt', header=None).values
    test = pd.read_csv('./data/select_data/jump/jump01.txt', header=None).values
    test = pd.read_csv('./data/select_data/light/light01.txt', header=None).values
    test = pd.read_csv('./data/select_data/run/slow_run01.txt', header=None).values
    test = pd.read_csv('./data/select_data/walk/normal_walk01.txt', header=None).values
    test = pd.read_csv('./data/select_data/upstairs/upstairs01.txt', header=None).values

    # get train data and label

    ' bicycle '
    path = './data/select_data/bicycle/'
    files = glob.glob(path + '*.txt')
    data_bicycle = []
    for file in files:
        data = pd.read_csv(file, header=None).values
        for i in range(len(data)):
            temp = data[i]
            data_bicycle.append(temp)
    label_bicycle = np.ones([1, len(data_bicycle)])[0] * 1

    ' downstairs '
    path = './data/select_data/downstairs/'
    files = glob.glob(path + '*.txt')
    data_downstairs = []
    for file in files:
        data = pd.read_csv(file, header=None).values
        for i in range(len(data)):
            temp = data[i]
            data_downstairs.append(temp)
    label_downstairs = np.ones([1, len(data_downstairs)])[0] * 2

    ' heavy '
    path = './data/select_data/heavy/'
    files = glob.glob(path + '*.txt')
    data_heavy = []
    for file in files:
        data = pd.read_csv(file, header=None).values
        for i in range(len(data)):
            temp = data[i]
            data_heavy.append(temp)
    label_heavy = np.ones([1, len(data_heavy)])[0] * 3

    ' jump '
    path = './data/select_data/jump/'
    files = glob.glob(path + '*.txt')
    data_jump = []
    for file in files:
        data = pd.read_csv(file, header=None).values
        for i in range(len(data)):
            temp = data[i]
            data_jump.append(temp)
    label_jump = np.ones([1, len(data_jump)])[0] * 4

    ' light '
    path = './data/select_data/light/'
    files = glob.glob(path + '*.txt')
    data_light = []
    for file in files:
        data = pd.read_csv(file, header=None).values
        for i in range(len(data)):
            temp = data[i]
            data_light.append(temp)
    label_light = np.ones([1, len(data_light)])[0] * 5

    ' run '
    path = './data/select_data/run/'
    files = glob.glob(path + '*.txt')
    data_run = []
    for file in files:
        data = pd.read_csv(file, header=None).values
        for i in range(len(data)):
            temp = data[i]
            data_run.append(temp)
    label_run = np.ones([1, len(data_run)])[0] * 6

    ' walk '
    path = './data/select_data/walk/'
    files = glob.glob(path + '*.txt')
    data_walk = []
    for file in files:
        data = pd.read_csv(file, header=None).values
        for i in range(len(data)):
            temp = data[i]
            data_walk.append(temp)
    label_walk = np.ones([1, len(data_walk)])[0] * 7

    ' upstairs '
    path = './data/select_data/upstairs/'
    files = glob.glob(path + '*.txt')
    data_upstairs = []
    for file in files:
        data = pd.read_csv(file, header=None).values
        for i in range(len(data)):
            temp = data[i]
            data_upstairs.append(temp)
    label_upstairs = np.ones([1, len(data_upstairs)])[0] * 8

    # conbine
    train = np.vstack((data_bicycle, data_downstairs, data_heavy, data_jump, data_light, data_run, data_walk, data_upstairs))
    label = np.hstack((label_bicycle, label_downstairs, label_heavy, label_jump, label_light, label_run, label_walk, label_upstairs))

########################################################################################################################

    # # logistic regression
    # prediction_lr, w, b = logistic_regression(train, label, test)
    #
    # # logistic regression result
    # for i in range(len(test)):
    #     s = np.dot(w, test[i])+b
    #     print("score: {}   Logistic Regression prediction: {}".format(s, prediction_lr[i]))
    # print('    ')
    # print('    ')
    # print('accuracy: {}'.format(np.mean(prediction_lr)))
    # print("after rounding accuracy: {}".format(train_data_old_version.rounding(np.mean(prediction_lr))))  # 四舍五入后的结果
    # # print(w,b)
    # # print(prediction_lr)

########################################################################################################################

    # random forest
    prediction_rf = random_forest(train, label, test)

    # random forest result
    for i in range(len(prediction_rf)):
        print("score:", prediction_rf[i])
    print('    ')
    print('    ')
    print('accuracy: {}'.format(np.mean(prediction_rf)))
    print("after rounding accuracy: {}".format(train_data_old_version.rounding(np.mean(prediction_rf))))  # 四舍五入后的结果
    # print(prediction_lr)
