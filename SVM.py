import numpy as np
import pandas as pd
from sklearn.svm import SVC

if __name__ == '__main__':

    # get test data
    test = pd.read_csv('./walk_data/heavy_light/feature_test_heavy.txt', header=None).values
    test_label = np.ones([1, len(test)])[0] * 1
########################################################################################################################
    # test_heavy = pd.read_csv('./walk_data/heavy_light/feature_test_heavy.txt', header=None).values
    # test_heavy_label = np.ones([1, len(test_heavy)])[0] * 1
    # test_light = pd.read_csv('./walk_data/heavy_light/feature_test_light.txt', header=None).values
    # test_light_label = np.ones([1, len(test_light)])[0] * 0
    #
    # test = np.vstack((test_light, test_heavy))
    # test_label = np.hstack((test_light_label, test_heavy_label))
########################################################################################################################
    # get train data
    data_heavy = pd.read_csv('./walk_data/heavy_light/feature_train_heavy.txt', header=None).values
    data_light = pd.read_csv('./walk_data/heavy_light/feature_train_light.txt', header=None).values
    # get train label
    label_heavy = np.ones([1, len(data_heavy)])[0] * 1  # 10000 == heavy
    label_light = np.ones([1, len(data_light)])[0] * 0  # -10000 == light
    # conbine
    label = np.hstack((label_heavy, label_light))
    train = np.vstack((data_heavy, data_light))

    # SVM
    svc = SVC(C=1.0, kernel='rbf', gamma='auto')
    print('Start training SVM...')
    svc.fit(train, label)
    print('Finished training')
    print('Testing...')
    # predict_svc = svc.predict(test)
    score = svc.score(test, test_label)
    print("The score of SVM is : %f" % score)