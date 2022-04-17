import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


# 这是一个随机的种子，来保证下面的结果将会是一致的
# np.random.seed(1)
# 将数据分别加载到变量X和Y中
X, Y = load_planar_dataset()
# 把数据集加载完毕了，可以使用plt对数据进行展示 其实c表示的是颜色的意思 s表示的是点的大小
plt.scatter(X[0,:], X[1,:], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
# plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]

print("x的维度是：" + str(shape_X))
print("y的维度是：" + str(shape_Y))
print("数据集里面的数据有：" + str(m) + "个")

# 在构建完整的神经网络之前，我们通过sklearn内置的函数，来看看逻辑回归在这个问题上面的表现如何
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

# 下面我们将逻辑回归分类器的分类绘制出来
plot_decision_boundary(lambda x: clf.predict(x), X, Y)  # 这是对决策边界的绘制
plt.title("lo re")
LR_predictions = clf.predict(X.T)
# 分子是计算出了预测正确的个数
print("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
       "% " + "(正确标记的数据点所占的百分比)")
# plt.show()

# 下面我们开始搭建神经网络
# 构建神经网络的一般方法是：定义神经网络的结构 初始化模型的参数 循环


# 下面开始定义神经网络
def layer_sizes(X, Y):
	"""
	:param X:  输入数据集，维度为（输入的数量，训练/测试的数量）
	:param Y: 标签
	:return: 输入层的数量，隐藏层的数量，输出层的数量
	"""
	n_x = X.shape[0]
	n_h = 6
	n_y = Y.shape[0]
	return n_x, n_h, n_y


# 测试layer_sizes
print("=========================测试layer_sizes=========================")
X_asses, Y_asses = layer_sizes_test_case()
n_x, n_h, n_y = layer_sizes(X_asses, Y_asses)
print("输入层的节点数量为: n_x = " + str(n_x))
print("隐藏层的节点数量为: n_h = " + str(n_h))
print("输出层的节点数量为: n_y = " + str(n_y))


# 初始化模型的参数
def initialize_parameters(n_x, n_h, n_y):
	"""
    parameters - 包含参数的字典：
            W1 - 权重矩阵,维度为（n_h，n_x）
            b1 - 偏向量，维度为（n_h，1）
            W2 - 权重矩阵，维度为（n_y，n_h）
            b2 - 偏向量，维度为（n_y，1）
    """
	# np.random.seed(2)	# 指定一个随机种子，以便你的输出与我们的一样。
	W1 = np.random.randn(n_h, n_x) * 0.01
	b1 = np.zeros(shape=(n_h, 1))
	W2 = np.random.randn(n_y, n_h) * 0.01
	b2 = np.zeros(shape=(n_y, 1))

	# 使用断言确保我的数据格式是正确的
	assert (W1.shape == (n_h, n_x))
	assert (b1.shape == (n_h, 1))
	assert (W2.shape == (n_y, n_h))
	assert (b2.shape == (n_y, 1))

	parameters = {"W1": W1,
				  "b1": b1,
				  "W2": W2,
				  "b2": b2}

	return parameters

# 循环
def forward_propagation(X, parameters):
	"""
    参数：
         X - 维度为（n_x，m）的输入数据。
         parameters - 初始化函数（initialize_parameters）的输出

    返回：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值
         cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型变量
     """
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	# 前向传播计算A2
	Z1 = np.dot(W1, X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)
	# 使用断言确保我的数据格式是正确的
	assert (A2.shape == (1, X.shape[1]))
	cache = {"Z1": Z1,
			 "A1": A1,
			 "Z2": Z2,
			 "A2": A2}

	return (A2, cache)


# 计算损失
def compute_cost(A2, Y, parameters):
	"""
    计算方程（6）中给出的交叉熵成本，

    参数：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值
         Y - "True"标签向量,维度为（1，数量）
         parameters - 一个包含W1，B1，W2和B2的字典类型的变量

    返回：
         成本 - 交叉熵成本给出方程（13）
    """

	m = Y.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]

	# 计算成本
	logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
	cost = - np.sum(logprobs) / m
	cost = float(np.squeeze(cost))

	assert (isinstance(cost, float))

	return cost


def backward_propagation(parameters, cache, X, Y):
	"""
    使用上述说明搭建反向传播函数。

    参数：
     parameters - 包含我们的参数的一个字典类型的变量。
     cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
     X - 输入数据，维度为（2，数量）
     Y - “True”标签，维度为（1，数量）

    返回：
     grads - 包含W和b的导数一个字典类型的变量。
    """
	m = X.shape[1]

	W1 = parameters["W1"]
	W2 = parameters["W2"]

	A1 = cache["A1"]
	A2 = cache["A2"]

	dZ2 = A2 - Y
	dW2 = (1 / m) * np.dot(dZ2, A1.T)
	db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
	dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
	dW1 = (1 / m) * np.dot(dZ1, X.T)
	db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
	grads = {"dW1": dW1,
			 "db1": db1,
			 "dW2": dW2,
			 "db2": db2}

	return grads


def update_parameters(parameters, grads, learning_rate=1.2):
	"""
    使用上面给出的梯度下降更新规则更新参数

    参数：
     parameters - 包含参数的字典类型的变量。
     grads - 包含导数值的字典类型的变量。
     learning_rate - 学习速率

    返回：
     parameters - 包含更新参数的字典类型的变量。
    """
	W1, W2 = parameters["W1"], parameters["W2"]
	b1, b2 = parameters["b1"], parameters["b2"]

	dW1, dW2 = grads["dW1"], grads["dW2"]
	db1, db2 = grads["db1"], grads["db2"]

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parameters = {"W1": W1,
				  "b1": b1,
				  "W2": W2,
				  "b2": b2}

	return parameters


def nn_model(X, Y, n_h, num_iterations, print_cost=False):
	"""
    参数：
        X - 数据集,维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
     """

	np.random.seed(3)  # 指定随机种子
	n_x = layer_sizes(X, Y)[0]
	n_y = layer_sizes(X, Y)[2]

	parameters = initialize_parameters(n_x, n_h, n_y)
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	for i in range(num_iterations):
		A2, cache = forward_propagation(X, parameters)
		cost = compute_cost(A2, Y, parameters)
		grads = backward_propagation(parameters, cache, X, Y)
		parameters = update_parameters(parameters, grads, learning_rate=0.5)

		if print_cost:
			if i % 1000 == 0:
				print("第 ", i, " 次循环，成本为：" + str(cost))
	return parameters


def predict(parameters, X):
	"""
    使用学习的参数，为X中的每个示例预测一个类

    参数：
        parameters - 包含参数的字典类型的变量。
        X - 输入数据（n_x，m）

    返回
        predictions - 我们模型预测的向量（红色：0 /蓝色：1）

     """
	A2, cache = forward_propagation(X, parameters)
	predictions = np.round(A2)

	return predictions



parameters = nn_model(X, Y, n_h = 6, num_iterations=10000, print_cost=True)

#绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

predictions = predict(parameters, X)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
