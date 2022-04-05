import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

# 解释一下这里的变量 第一个存放的是训练集里面的数据，第二个是存放的是训练集的标签
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# 下面开始分别存放了照片的数量
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

print ("训练集的数量: m_train = " + str(m_train))
print ("测试集的数量 : m_test = " + str(m_test))
print ("每张图片的宽/高 : num_px = " + str(num_px))
print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
print ("测试集_标签的维数: " + str(test_set_y.shape))

# 将训练集的维度降低并且转置，这个里面的的矩阵的大小在转置之前应该是209，64*64*3
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并且转置
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 这个里面主要采用的思想是希望能够将参数控制到一个可控的范围内，同时运用到了广播的概念
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


# 下面开始定义sigmoid函数，也就是激活函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# 下面是对w, b 进行初始化的函数
# 参数的意思是希望为函数创建一个(dim, 1)的0向量，并且把b初始化成0
# 参数dim也就是w的维度
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    return w, b


# 下面是一个去计算cost的函数,参数主要的含义是w权重，b偏置
def propagate(w, b, X, Y):
    # 这里的m应该表示的是图片的数量
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    # 这个地方是按照公式进行推获得
    dz = A - Y
    dw = (1/m) * np.dot(X, dz.T)
    db = (1/m) * np.sum(dz)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return dw, db, cost

# 测试一下上面的函数
# print("测试")
# w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
# dw, db = propagate(w, b, X, Y)
# print(dw)
# print(db)


# 下面使用渐变下降更新参数，参数的含义前四个不用多说，第五个是学习的次数，第六个是学习率，第七个是决定是否要进行打印
def optimize(w, b, X, Y, num, learning_rate):
    for i in range(num):
        dw, db, cost = propagate(w, b, X, Y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            print("迭代的次数为：%i" % i)
    return w, b


# 下面一个便是预测的函数
def predict(w, b, X):
    m = X.shape[1]  # 获取图片的数量
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 计预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # 将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    # 使用断言
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):
    """
    通过调用之前实现的函数来构建逻辑回归模型

    参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本

    返回：
        d  - 包含有关模型信息的字典。
    """
    w, b = initialize_with_zeros(X_train.shape[0])

    w, b = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    # 预测测试/训练集的例子
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    d = {
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d


print("====================测试model====================")
# 这里加载的是真实的数据，请参见上面的代码部分。
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005)
