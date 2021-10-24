import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import glob
import os

import setting


if __name__ == '__main__':
    new_folder = 'CNN'
    old_folder = 'data_cnn' #'data_for_testing' #'data_rnn'
    train_num = 200
    test_num = 50


    new_path = './data/' + new_folder + '/'
    old_path = './data/' + old_folder + '/'
    os.mkdir(new_path) # 创建新的主文件夹
    os.mkdir(new_path + 'train')  # 创建新的train文件夹
    os.mkdir(new_path + 'test')  # 创建新的test文件夹

    type_list = setting.get_type()
    for type in type_list:
        new_sub_train_folder = new_path + 'train/' + type
        new_sub_test_folder = new_path + 'test/' + type
        os.mkdir(new_sub_train_folder) # 创建新的次文件夹
        os.mkdir(new_sub_test_folder)  # 创建新的次文件夹

        old_files_path = old_path + type + '/'
        old_files = glob.glob(old_files_path + '*.txt')
        new_train_path = new_path + 'train/' + type + '/'
        new_test_path = new_path + 'test/' + type + '/'
        # new_files = glob.glob(new_files_path + '*.txt')

        for i in range(train_num):
            num = i +1
            oldfile = old_files_path + type + str(num) + '.txt'
            shutil.copy(oldfile, new_train_path)  # oldfile 只能是文件夹，newfile 可以是文件，也可以是目标目录

        if len(old_files) - train_num >= test_num:
            var = test_num
        else:
            var = len(old_files) - train_num

        for i in range(var):
            num = i + train_num + 1
            oldfile = old_files_path + type + str(num) + '.txt'
            shutil.copy(oldfile, new_test_path)