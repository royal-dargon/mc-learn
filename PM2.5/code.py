# 训练集中的数据的排列符合人类的观察习惯，但是不能直接拿给模型去训练
# 值得注意的是RainFall那一栏表示的对应的时段是否下雨，如果下雨就是1，如果没有下雨就是NR，因此在采用补全处理的方式，将NR变成0就OK
# 这个地方对于每一天的信息维度是(18,24)十八个测量维度，二十四个时间节点
# 由于需要得到240条不重复的测试数据，可以选择把一天开始的九个小时作为训练数据，然后把九点的数据的PM2.5提取出来变成结果，以此类推一天可以得到十五个这样的结果

import pandas as pd
import numpy as np


# 数据预处理阶段
def dataProcess(df):
    x_list, y_list = [], []
    df = df.replace(['NR'], [0.0])
    # df = df.iloc[:, 3:]
    # df[df == 'NR'] = 0
    # df = df.to_numpy()
    array = np.array(df).astype(float)
    # 将数据拆分为多个数据帧
    for i in range(0, 4320, 18):
        for j in range(24-9):
            mat = array[i:i+18, j:j+9]
            label = array[i+9, j+9]  # 第十行是PM2.5
            x_list.append(mat)
            y_list.append(label)
    x = np.array(x_list)  # x相当于训练的数
    y = np.array(y_list)  # y相当于训练最终的标签
    return x, y, array


# 开始训练
def train(x_train, y_train, epoch):
    bias = 0  # 这是最初始的偏差值
    weights = np.ones(9)  # 初始化权重，前九个的权重是相同的
    learning_rate = 1  # 初始化学习率
    reg_rate = 0.001  # 正则化系数
    bg2_sum = 0  # 用来存放偏差值的梯度平方和
    wg2_sum = np.zeros(9)  # 用来存放权重的梯度平方和

    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(9)
        # 在所有数据上计算Loss的梯度
        for j in range(3200):
            b_g += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-1)
            for k in range(9):
                w_g[k] += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-x_train[j, 9, k])
        # 求平均值
        b_g /= 3200
        w_g /= 3200
        # 加上正则化的梯度
        for m in range(9):
            w_g[m] += reg_rate * weights[m]

        bg2_sum += b_g**2
        wg2_sum += w_g**2
        # 更新权重和偏移
        bias -= learning_rate/bg2_sum**0.5 * b_g
        weights -= learning_rate/wg2_sum**0.5 * w_g

        # 每次训练200轮就输出一次结果
        if i%200 == 0:
            loss = 0
            for j in range(3200):
                loss += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias)**2
            print('after {} epochs,the loss on train data is:'.format(i), loss/3200)

    return weights, bias


# 验证模型效果
def validate(x_val, y_val, weights, bias):
    loss = 0
    for i in range(400):
        loss += (y_val[i] - weights.dot(x_val[i, 9, :]) - bias)**2
    return loss/400


def main():
    # 从csv中获得有效的信息
    df = pd.read_csv('train.csv', usecols=range(3, 27), encoding='gb18030')
    # df.replace(['NR'], [0.0])
    # print(df)
    x, y, _ = dataProcess(df)
    # 划分训练集与验证集
    x_train, y_train = x[0:3200], y[0:3200]
    x_test, y_test = x[3200:3600], y[3200:3600]  # 这里的3600来自于一共240天，每天可以产生15个训练集，最终划分
    epoch = 2000  # 训练的轮数
    # 开始训练
    w, b = train(x_train, y_train, epoch)
    # 在验证集上面看效果怎么样
    loss = validate(x_test, y_test, w, b)
    print('the loss on val data is:', loss)


if __name__ == '__main__':
    main()