#   用于获得波峰波谷
#   其中 wave_double 用的是传统的方式来进行波峰波谷的识别
#   wave_diff 则是使用差分的方式，寻找两个波之间的平缓部分来进行识别

import numpy as np

'peak_valley'
def wave_double(data, thh):
    MAX = 2
    MIN = 1
    p1 = []
    p2 = []
    p3 = []
    waves = []
    type_point = 0
    max_point = -np.inf
    min_point = np.inf
    max_position = 0
    min_position = 0
    count = 0

    for i in range(len(data)):
        if data[i] > max_point:
            max_point = data[i]
            max_position = i
        if data[i] < min_point:
            min_point = data[i]
            min_position = i
        if type_point == 0:
            if max_point - min_point >= thh:
                if max_position > min_position:
                    p1.append(min_point)
                    p1.append(min_position)
                    p1.append(MIN)
                    type_point = 1
                else:
                    type_point = 0
                max_point = -np.inf
                min_point = np.inf
                max_position = i
                min_position = i
        elif type_point == 1:
            if max_point - min_point >= thh:
                isvalid = 0
                if max_position > min_position and p1[2] == MAX:  # not gonna happen
                    p2.append(min_point)
                    p2.append(min_position)
                    p2.append(MIN)
                    isvalid = 1
                if max_position < min_position and p1[2] == MIN:
                    p2.append(max_point)
                    p2.append(max_position)
                    p2.append(MAX)
                    isvalid = 1
                if isvalid == 1:
                    type_point = 2
                max_point = -np.inf
                min_point = np.inf
                max_position = i
                min_position = i
        elif type_point == 2:
            if max_point - min_point >= thh:
                isvalid = 0
                if max_position > min_position and p2[2] == MAX:
                    h1 = p2[0] - min(data[p1[1]:p2[1]])
                    h2 = abs(p2[0] - min_point)
                    if h1 > h2 and (h2 / h1) < 0.6:
                        max_point = -np.inf
                        min_point = np.inf
                        max_position = i
                        min_position = i
                        continue
                    if min_position-p1[1] <= 80:
                        max_point = -np.inf
                        min_point = np.inf
                        max_position = i
                        min_position = i
                        type_point = 1
                        p2 = []
                        continue
                    p3.append(min_point)
                    p3.append(min_position)
                    p3.append(MIN)
                    isvalid = 1
                if max_position < min_position and p2[2] == MIN:
                    h1 = p2[0] - min(data[p1[1]:p2[1]])
                    h2 = abs(p2[0] - min_point)
                    if h1 > h2 and (h2 / h1) < 0.6:
                        max_point = -np.inf
                        min_point = np.inf
                        max_position = i
                        min_position = i
                        continue
                    if min_position-p1[1] <= 20:
                        max_point = -np.inf
                        min_point = np.inf
                        max_position = i
                        min_position = i
                        type_point = 1
                        p2 = []
                        continue
                    p3.append(min_point)
                    p3.append(min_position)
                    p3.append(MIN)
                    isvalid = 1
                max_point = -np.inf
                min_point = np.inf
                max_position = i
                min_position = i
                if isvalid == 1:
                    type_point = 1
                    count = count + 1
                    # 修正P2点的位置
                    for k in range(p1[1], p3[1]):
                        if data[k] > p2[0]:
                            p2[0] = data[k]
                            p2[1] = k
                    waves.append([p1[1], p2[1], p3[1]])
                    p1 = p3
                    p2 = []
                    p3 = []
                    continue
            if i-p1[1] >= 2000 and i-p2[1] >= 1000:
                Y = min(np.array(data)[p2[1]:i])
                I = np.argmin(np.array(data)[p2[1]:i])
                if p2[0]-Y >= thh:
                    p3.append(Y)
                    p3.append(p2[1] - 1 + I)
                    p3.append(MIN)
                    type_point = 0
                    max_point = -np.inf
                    min_point = np.inf
                    max_position = i
                    min_position = i
                    count = count + 1
                    waves.append([p1[1], p2[1], p3[1]])
                    p1 = p3
                    p2 = []
                    p3 = []

    return waves


'peak valley diff'
def wave_diff(data):
    num_diff = 19  # 差分片段的长度(奇数)
    total_diff = 500  # 差分满足的条件
    p_list = []  # 存放临时的p点
    waves = []  # 存放最后的p点
    type_point = 0
    rows = len(data)
    for i in range(rows - num_diff - 1):  # 抛掉前n-1个 以后 倒数n-1个
        current_p = i
        temp = data[i + int((num_diff - 1) / 2)]
        diff_list = []
        # 选取该点之后2n+2个点的段落
        for j in range(num_diff):
            minus = data[i + j] - temp
            diff_list.append(abs(minus))

        # 寻找起始点
        if type_point == 0:
            if sum(diff_list) <= total_diff:  # 满足条件
                if current_p == 0:
                    p_list.append(current_p)
                else:
                    p_list.append(current_p)
                    last_p = current_p
            else:
                isvalue = 1
                if len(p_list) >= 50:
                    for k in range(len(p_list) - 1):
                        if p_list[k + 1] - p_list[k] != p_list[1] - p_list[0]:
                            isvalue = 0
                            p_list = []
                else:
                    isvalue = 0
                    p_list = []
                # 找到起始点
                if isvalue == 1:
                    waves.append(last_p)
                    p_list = []
                    type_point = 1

        if type_point == 1:
            if sum(diff_list) <= total_diff:  # 满足条件
                p_list.append(current_p)
                last_p = current_p
            else:
                isvalue = 1
                if len(p_list) >= 20:
                    for k in range(len(p_list) - 1):
                        if p_list[k + 1] - p_list[k] != p_list[1] - p_list[0]:
                            isvalue = 0
                            p_list = []
                else:
                    isvalue = 0
                    p_list = []
                # 找到下一个节点
                if isvalue == 1:
                    if len(p_list) % 2 == 0:
                        p = p_list[int(len(p_list) / 2)]
                    else:
                        p = p_list[int(len(p_list) / 2) + 1]
                    waves.append(p)
                    p_list = []
    return waves
