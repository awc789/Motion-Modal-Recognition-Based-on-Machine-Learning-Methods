import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable
from sklearn import preprocessing

from rnn_prepare import set_data_9, set_data_10


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=6,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

def read_data(f):
    data = pd.read_csv(f)
    data['date'] = range(len(data))
    data.columns = ['count', 'date']
    # data = data.set_index('date')
    # data.index = pd.to_datetime(data.index)
    ts = data['count']
    # draw_ts(ts)  # 绘图
    return ts

if __name__ == '__main__':
    torch.manual_seed(1)    # reproducible

    # Hyper Parameters
    TIME_STEP = 10      # rnn time step
    INPUT_SIZE = 1      # rnn input size
    LR = 0.01           # learning rate
    length = 0          # 绘图用长度

    # show data
    file_9 = './data/data_after_kalman/sample9/bicycle/bicycle01.txt'
    file_10 = './data/data_after_kalman/sample10/bicycle/bicycle01.txt'
    # 用9轴数据预测10轴数据
    freq, waves, test_size = set_data_10(file_9, file_10)
    filename_9 = './data/cycle_9.txt'
    filename_10 = './data/cycle_10.txt'
    ts_9 = read_data(filename_9).values # 读取数据
    ts_10 = read_data(filename_10).values

    # 标准化
    ts_9 = preprocessing.scale(ts_9) * 10
    ts_10 = preprocessing.scale(ts_10) * 10

    # 绘图
    # plt.subplot(211)
    # plt.plot(ts_9)
    # plt.subplot(212)
    # plt.plot(ts_10)



    rnn = RNN().double()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.MSELoss()

    h_state = None      # for initial hidden state

    plt.figure(1, figsize=(12, 5))
    plt.ion()           # continuously plot

    for step in range(len(waves)):
        start, end = waves[step][0], waves[step][2]   # time range
        # use sin predicts cos
        # x_np = ts_9[start:end]    # float32 for converting torch FloatTensor
        x_np = ts_10[start:end]
        y_np = ts_10[start:end]

        x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
        y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

        prediction, h_state = rnn(x, h_state)   # rnn output
        # !! next step is important !!
        h_state = Variable(h_state.data)        # repack the hidden state, break the connection from last iteration

        loss = loss_func(prediction, y)         # cross entropy loss
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients


        # plotting
        plt.plot(range(length, length + len(y_np.flatten())), y_np.flatten(), 'r-')
        plt.plot(range(length, length + len(y_np.flatten())), prediction.data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.05)

        length = length + len(y_np.flatten())