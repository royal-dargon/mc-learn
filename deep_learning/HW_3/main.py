import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


# 这是一个随机的种子，来保证下面的结果将会是一致的
np.random.seed(1)
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
plt.show()

# 下面我们开始搭建神经网络

