import pandas as pd
import numpy as np
import math

data = pd.read_csv('train.csv', encoding='gb18030')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0

raw_data = data.to_numpy()
month_data = {}
# 数据初步转换，拼接了每一天
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24:(day + 1) * 24] = raw_data[18 * (20 * month + day):18*(month * 20 + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)

for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            # 完成了采样的工作
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day*24+hour:day*24+hour+9].reshape(1, -1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day*24+hour+9]

# 标准化
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)

for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

x_train_set = x[:math.floor(len(x) * 0.8), :]
y_train_set = y[:math.floor(len(x) * 0.8), :]

x_test_set = x[math.floor(len(x) * 0.8):, :]
y_test_set = y[math.floor(len(x)*0.8):, :]

# 设置维度，1是b
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x_train_set = np.concatenate((np.ones([len(x_train_set), 1]), x_train_set), axis=1).astype(float)

# 学习率与循环次数
learning_rate = 100
iter_time = 3000

# print(len(x_train_set))
adagrad = np.zeros([dim, 1])
eps = 0.0001
a_sum = np.zeros([dim, 1])
for t in range(iter_time):
    loss = np.sqrt(sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))/len(x_train_set))
    if(t % 100 == 0):
        print("迭代次数：%i 损失: %f" %(t, loss))
    temp = np.dot(x_train_set, w)
    # print(y_train_set.T.shape)
    # adagrad = np.dot(np.dot(w.T, x_train_set.T), np.dot(x_train_set, w) - y_train_set)/(loss * len(x_train_set))
    adagrad = np.dot(np.dot(x_train_set.T, x_train_set), w) - np.dot(x_train_set.T, y_train_set)/len(x_train_set)

    # print(adagrad.shape)
    g = adagrad
    a_sum += (adagrad * adagrad)
    # print(adagrad.shape)
    # 梯度下降
    w = w - (learning_rate * g) / (np.sqrt(a_sum) + eps)
    # print(w)
np.savetxt('weights.txt', w)
