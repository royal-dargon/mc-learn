import math

import numpy as np
import pandas as pd


def get_date(file_name):
    data = pd.read_csv(file_name)
    # 观察数据的结构对我们需要的数据进行收集
    data_set = data.iloc[:, 3:]
    # 对数据中出现的NR数据进行数据的处理
    data_set[data == 'NR'] = 0
    # 在这里我们对训练集的收集主要是一个按照月为单位的收集方式，每个月连续的二十天，一共是480条的数据，然后通过窗口移动的方式每个月可以出现481条
    month_list = {}
    raw_data = np.array(data_set).astype(float)
    for month in range(12):
        temp = np.empty([18, 480])
        for day in range(20):
            temp[:, day * 24:day * 24 + 24] = raw_data[(month * 20 + day) * 18:(month * 20 + day + 1) * 18, :]
        month_list[month] = temp
        # print(temp)
    return month_list


# 进行训练集与测试集进行划分的函数
def get_set(month_list):
    # 下面是输入的数据
    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                # 在这里便是完成了训练集和他们对应的标签
                x[month * 471 + day * 24 + hour, :] = month_list[month][:, day * 24 + hour:day * 24 + hour + 9].reshape(1, 9 * 18)
                y[month * 471 + day * 24 + hour, 0] = month_list[month][9, day * 24 + hour + 9]
    # 下面便是将数据进行标准化的处理,这里的axis主要是实现了行的压缩，求各列的均值或者是标准差
    x_std = np.std(x, axis=0)
    x_mean = np.mean(x, axis=0)
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x_std[j] != 0:
                x[i][j] = (x[i][j] - x_mean[j]) / x_std[j]
    x_train = x[:math.floor(len(x) * 0.8), :]
    y_train = y[:math.floor(len(y) * 0.8), 0]
    # print(y_train)
    return x_train, y_train


# 模型的训练
def train(x_train, y_train, times):
    pass


def main():
    # 开始对数据获取并且进行处理的工作
    month_list = get_date('train.csv')
    # 将上一步从文件中获取的数据传入到我们对训练与测试进行划分的函数中
    x_train, y_train = get_set(month_list)
    # 设置循环的次数
    times = 2000
    train(x_train, y_train, times)


if __name__ == '__main__':
    main()