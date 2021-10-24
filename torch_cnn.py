import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import glob
from matplotlib import cm
import random
from sklearn import preprocessing
from sklearn.svm import SVC


''' memo '''
# import torch
# import random
#
# if __name__ == '__main__':
#
#     label_0, label_1 = np.zeros([1, 1000])[0], np.ones([1, 1000])[0]
#     test_label = np.hstack((label_0, label_1))
#     test_label = torch.tensor(test_label)
#
#     num_0, num_1 = 0, 0
#
#     for i in range(len(test_label)):
#         if test_label[i] == 0:
#             num_0 += 1
#         elif test_label[i] == 1:
#             num_1 += 1
#
#     print('Num1:  ', num_1)
#     print('Num0:  ', num_0)
#
#     random.shuffle(test_label)
#
#     num_0, num_1 = 0, 0
#     for i in range(len(test_label)):
#         if test_label[i] == 0:
#             num_0 += 1
#         elif test_label[i] == 1:
#             num_1 += 1
#
#     print('After Shuffle Num1:  ', num_1)
#     print('After Shuffle Num0:  ', num_0)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 44, 200)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=32,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 44, 200)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 22, 100)
        )
        self.conv2 = nn.Sequential(         # input shape (32, 22, 100)
            nn.Conv2d(32, 64, 3, 1, 1),     # output shape (64, 22, 100)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (64, 11, 50)
        )
        self.out = nn.Linear(64 * 11 * 50, 10)   # fully connected layer, output 8 classes (but use 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 64 * 11 * 55)
        output = self.out(x)
        return output, x    # return x for visualization


def train_data_function(files, train_data, train_label, n):

    for file in files:
        var_train = np.array(pd.read_csv(file, header=None).values)
        var_train = preprocessing.scale(var_train) * 100
        train_data.append([var_train])
        train_label.append(n)

    return train_data, train_label


def test_data(files, test, test_label, n):

    for file in files:
        var_test = np.array(pd.read_csv(file, header=None).values)
        var_test = preprocessing.scale(var_test) * 100
        test.append([var_test])
        test_label.append(n)

    return test, test_label


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)


########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == '__main__':

    # Hyper Parameters
    EPOCH = 5                # train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE = 50         # batch size
    LR = 0.0001              # learning rate

    # 2/10/0.001

    ''' get data address '''
    ''' train '''
    # bicycle
    path_bicycle = './data/CNN/train/bicycle/'
    files_bicycle = glob.glob(path_bicycle + '*.txt')
    # downstairs
    path_downstairs = './data/CNN/train/downstairs/'
    files_downstairs = glob.glob(path_downstairs + '*.txt')
    # heavy
    path_heavy = './data/CNN/train/heavy/'
    files_heavy = glob.glob(path_heavy + '*.txt')
    # jump
    path_jump = './data/CNN/train/jump/'
    files_jump = glob.glob(path_jump + '*.txt')
    # light
    path_light = './data/CNN/train/light/'
    files_light = glob.glob(path_light + '*.txt')
    # run
    path_run = './data/CNN/train/run/'
    files_run = glob.glob(path_run + '*.txt')
    # upstairs
    path_upstairs = './data/CNN/train/upstairs/'
    files_upstairs = glob.glob(path_upstairs + '*.txt')
    # walk
    path_walk = './data/CNN/train/walk/'
    files_walk = glob.glob(path_walk + '*.txt')


    ''' test '''
    # test bicycle
    path_test_bicycle = './data/CNN/test/bicycle/'
    files_test_bicycle = glob.glob(path_test_bicycle + '*.txt')
    # test downstairs
    path_test_downstairs = './data/CNN/test/downstairs/'
    files_test_downstairs = glob.glob(path_test_downstairs + '*.txt')
    # test heavy
    path_test_heavy = './data/CNN/test/heavy/'
    files_test_heavy = glob.glob(path_test_heavy + '*.txt')
    # test jump
    path_test_jump = './data/CNN/test/jump/'
    files_test_jump = glob.glob(path_test_jump + '*.txt')
    # test light
    path_test_light = './data/CNN/test/light/'
    files_test_light = glob.glob(path_test_light + '*.txt')
    # test run
    path_test_run = './data/CNN/test/run/'
    files_test_run = glob.glob(path_test_run + '*.txt')
    # test upstairs
    path_test_upstairs = './data/CNN/test/upstairs/'
    files_test_upstairs = glob.glob(path_test_upstairs + '*.txt')
    # test walk
    path_test_walk = './data/CNN/test/walk/'
    files_test_walk = glob.glob(path_test_walk + '*.txt')


    ''' get data '''
    train_data, train_label = [], []
    test, test_label = [], []

    ''' bicycle  --------------------  1 '''
    ''' downstairs  -----------------  2 '''
    ''' heavy  ----------------------  3 '''
    ''' jump  -----------------------  4 '''
    ''' light  ----------------------  5 '''
    ''' run  ------------------------  6 '''
    ''' upstairs  -------------------  7 '''
    ''' walk  -----------------------  8 '''

    # train
    print('Strat to get train data ... ')
    train_data, train_label = train_data_function(files_bicycle, train_data, train_label, 1)
    train_data, train_label = train_data_function(files_downstairs, train_data, train_label, 2)
    train_data, train_label = train_data_function(files_heavy, train_data, train_label, 3)
    train_data, train_label = train_data_function(files_jump, train_data, train_label, 4)
    train_data, train_label = train_data_function(files_light, train_data, train_label, 5)
    train_data, train_label = train_data_function(files_run, train_data, train_label, 6)
    train_data, train_label = train_data_function(files_upstairs, train_data, train_label, 7)
    train_data, train_label = train_data_function(files_walk, train_data, train_label, 8)
    # shuffle the data
    randnum_train = random.randint(0, len(train_data))
    random.seed(randnum_train)
    random.shuffle(train_data)
    random.seed(randnum_train)
    random.shuffle(train_label)
    # transform to tensor
    train_data, train_label = np.array(train_data), np.array(train_label)
    print('************************************************************************************************')

    # test
    print('Strat to get test data ... ')
    test, test_label = test_data(files_test_bicycle, test, test_label, 1)
    test, test_label = test_data(files_test_downstairs, test, test_label, 2)
    test, test_label = test_data(files_test_heavy, test, test_label, 3)
    test, test_label = test_data(files_test_jump, test, test_label, 4)
    test, test_label = test_data(files_test_light, test, test_label, 5)
    test, test_label = test_data(files_test_run, test, test_label, 6)
    test, test_label = test_data(files_test_upstairs, test, test_label, 7)
    test, test_label = test_data(files_test_walk, test, test_label, 8)

    # shuffle the data
    randnum_test = random.randint(0, len(test))
    random.seed(randnum_test)
    random.shuffle(test)
    random.seed(randnum_test)
    random.shuffle(test_label)
    # transform to tensor
    test, test_label = torch.tensor(np.array(test)), torch.tensor(np.array(test_label))
    print('************************************************************************************************')

    ''' CNN '''
    print('CNN')
    cnn = CNN()
    print(cnn)  # net architecture
    cnn = cnn.float()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()


    try:
        from sklearn.manifold import TSNE;
        HAS_SK = True
    except:
        HAS_SK = False;
        print('Please install sklearn for layer visualization')

    # set data loader
    # train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  # 打乱数据
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)  # 不打乱数据

    plt.ion()
    for epoch in range(EPOCH):
        for step, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            k = step * BATCH_SIZE
            label = torch.tensor(train_label[k: k + BATCH_SIZE])

            output = cnn(data.float())[0]  # cnn output
            loss = loss_func(output, label)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 10 == 0:
                test_output, last_layer = cnn(test.float())
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_label.data.numpy()).astype(int).sum()) / float(test_label.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
                if HAS_SK:
                    # Visualization of trained flatten layer (T-SNE)
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                    labels = test_label.numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels)
        plt.ioff()

        # print 10 predictions from test data
        num = np.random.randint(0, high=len(test), size=10, dtype='l')
        predict_result = []
        real_result = []
        for i in range(len(num)):
            test_output, _ = cnn(test[num[i] : num[i]+1].float())
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            predict_result.append(int(pred_y))
            real_result.append(int(test_label[num[i]].numpy()))

        print(' ')
        print(('get 10 random number'))
        print(predict_result, 'prediction number')
        print(real_result, 'real number')
        print(' ')

    # torch.save(cnn, 'cnn_model.pkl')
    # model = torch.load('cnn_model.pkl')

########################################################################################################################

    # SVM
    SVM_train_data, SVM_train_label = [], []
    SVM_test_data, SVM_test_label = [], []
    # get data
    # train
    cnn_output, cnn_layer = cnn(torch.tensor(train_data).float())
    for i in range(len(cnn_layer)):
        var = cnn_layer[i].detach().numpy()
        SVM_train_data.append(var)
        SVM_train_label.append(train_label[i])
    # test
    test_output, test_layer = cnn(test.float())
    for i in range(len(test_layer)):
        var = test_layer[i].detach().numpy()
        SVM_test_data.append(var)
        SVM_test_label.append(test_label[i].numpy())

    # train modle
    # linear：线性核函数 ｜ poly：多项式核函数 ｜ rbf：径像核函数 / 高斯核 ｜ sigmod：sigmod 核函数 ｜ precomputed：核矩阵
    # 'kernel=poly' --->  degree = n 指多项式核函数的阶数 n
    svc = SVC(C=1.0, kernel='rbf', gamma='auto')
    print('Start training SVM...')
    svc.fit(SVM_train_data, SVM_train_label)
    print('Finished training')
    print('Testing...')
    # predict_svc = svc.predict(SVM_test_data)
    # accuracy = float((predict_svc == test_label.data.numpy()).astype(int).sum()) / float(test_label.size(0))
    score = svc.score(SVM_test_data, SVM_test_label)
    print("The score of SVM is : %f" % score)

########################################################################################################################
    # # T-SNE
    # tsne = TSNE(perplexity=30, n_components=1, init='pca', n_iter=5000)
    # low_dim_embs = tsne.fit_transform(test_layer.data.numpy())
    # labels = test_label.numpy()
    #
    # import matplotlib.pyplot as plt
    # plt.plot(range(len(labels)), labels, 'bo', range(len(low_dim_embs)), low_dim_embs, 'r--')
    # plt.show()









