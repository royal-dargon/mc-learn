import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# 数据预处理阶段
def dataProcess(data):
    X_list, Y_list = [], []
    # data.replace(['Iris-setosa'], [0.0])
    # data.replace(['Iris-versicolor'], [1.0])
    # data.replace(['Iris-virginica'], [2.0])
    # print(data.shape)
    labelencoder = LabelEncoder()
    array = np.array(data)
    # 这里是将文字类型的结果进行数字的输出
    array[:, 4] = labelencoder.fit_transform(array[:, 4])
    array = np.array(array).astype(float)
    # 将每一行的一到四列加到x中，将第五列加到y中
    for i in range(150):
        x = array[i][:4]
        y = array[i][4]
        X_list.append(x)
        Y_list.append(y)
    X_list = np.array(X_list)
    Y_list = np.array(Y_list)
    # show(X_list, Y_list)
    return X_list, Y_list


# 定义一个sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))


# 定义一个损失函数, 这里就不考虑正则化了
def cost(w, x, y):
    first = -y * np.log(sigmoid(np.dot(w.T, x.T)))
    second = (1 - y) * np.log((1 - sigmoid(np.dot(w.T, x.T))))
    hx = sigmoid(np.dot(w.T, x.T))
    # print(hx)
    gradient = np.dot((hx - y), x) / len(x)
    # print(gradient.shape)
    return np.sum(first - second) / len(x), gradient.T


def train(x, y, epoch):
    # 一共有四个权重
    weights = np.ones([4, 1])
    learnRate = 0.001
    for i in range(epoch):
        loss, gre = cost(weights, x, y)
        # if i % 100 == 0:
        #     print("第"+str(i)+"次："+str(loss))
            # print(weights)
        weights += -learnRate * gre
        # print(weights)
    return weights


# 这是一个对数据进行可视化处理的函数
# def show(X_train, y_train):
#     fig, ax = plt.subplots(3, 3, figsize=(15, 15))
#     plt.suptitle("iris_pairplot")
#     for i in range(3):
#         for j in range(3):
#             ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
#             ax[i, j].set_xticks(())
#             ax[i, j].set_yticks(())
#     plt.show()

# 这个是一个测试的函数,我的思路是对三类参数进行测试
def test(w1, w2, w3, x, y):
    pre1 = sigmoid(np.dot(w1.T, x.T))
    pre2 = sigmoid(np.dot(w2.T, x.T))
    pre3 = sigmoid(np.dot(w3.T, x.T))
    print(pre3)


def main():
    data = pd.read_csv('iris.data', header=None)
    # print(data)
    x, y = dataProcess(data)
    x_train, y_train = x[:150], y[:150]
    # print(len(x_train))
    x_test, y_test = [], []
    epoch = 1000
    y_train0 = list(y_train)
    for i in range(40, 50):
        x_test.append(x_train[i])
        y_test.append(y_train[i])
    for i in range(90, 100):
        x_test.append(x_train[i])
        y_test.append(y_train[i])
    for i in range(140, 150):
        x_test.append(x_train[i])
        y_test.append(y_train[i])
    # print(x_test)
    for i in range(0, 50):
        y_train0[i] = 1
    for i in range(50, 150):
        y_train0[i] = 0

    y_train1 = list(y_train)
    for i in range(0, 50):
        y_train1[i] = 0
    for i in range(100, 150):
        y_train1[i] = 0
    y_train2 = list(y_train)
    for i in range(0, 100):
        y_train2[i] = 0
    for i in range(100, 150):
        y_train2[i] = 1
    w1 = train(x_train, np.array(y_train0), epoch)
    w2 = train(x_train, np.array(y_train1), epoch)
    w3 = train(x_train, np.array(y_train2), epoch)
    test(w1, w2, w3, np.array(x_test), np.array(y_test))


if __name__ == "__main__":
    main()