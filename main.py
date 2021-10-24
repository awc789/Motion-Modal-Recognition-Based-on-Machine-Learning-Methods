import numpy as np
import setting
import get_feature
import glob
import time

if __name__ == "__main__":
    start = time.clock()
    file_list = setting.get_type()
    path_10 = './data/sample10/'
    path_9 = './data/sample9/'

    path_after_kalman = './data/data_after_kalman/'

    for type in file_list:
        number = 0
        files = glob.glob(path_10 + type + '/*.txt')

        # check after kalman data
        files_after_kalman = glob.glob(path_after_kalman + 'sample10/' + type + '/*.txt')
        after_kalman_names = []
        for file_after_kalman in files_after_kalman:
            names = file_after_kalman.split('/')
            after_kalman_names.append(names[-1])

        for file in files:
            file_name = file.split('_')[-1]
            if file_name in after_kalman_names: # 存在 kalman filter 后的数据
                data_type = 1
                file_10 = path_after_kalman + 'sample10/' + type + '/' + file_name
                file_9 = path_after_kalman + 'sample9/' + type + '/' + file_name
            else: # 需要进行 kalman filter
                data_type = 0
                file_10 = path_10 + type + '/sample10_' + file_name
                file_9 = path_9 + type + '/sample9_' + file_name

            # start program
            print('************************************************************************************************')
            print('type: {} , file name: {}'.format(type, file_name))
            features = get_feature.feature_overall(file_9, file_10, data_type, type, file_name)

            for file_number in range(len(features)):
                number = number + 1
                feature = features[file_number]
                np.savetxt('./data/data_feature/' + type + '/' + type + str(number) + '.txt',
                       feature, fmt="%f", delimiter=',')

            np.savetxt('./data/select_data_total/' + type + '/' + file_name,
                       features, fmt="%f", delimiter=',')
            print('************************************************************************************************')

    elapsed = (time.clock() - start)
    hour = int(elapsed // 3600)
    minute = int((elapsed % 3600) // 60)
    second = int(elapsed - 3600 * hour - 60 * minute)
    print(' ')
    print(' ')
    print(' ')
    print("Time used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))