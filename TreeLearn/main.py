# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from sklearn import tree

# datasets 自带的数据
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
import graphviz

wine = load_wine()
res = pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)

name = wine.feature_names

# 百分之三十是测试集，百分之七十是测试集
# 需要注意前面的数据
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)

# 实例化
clf = tree.DecisionTreeClassifier(criterion="entropy")

# 进行训练的接口
clf = clf.fit(Xtrain,Ytrain)

# 进行打分操作
score = clf.score(Xtest,Ytest)

print(score)

# 下面开始画一棵树

# 参数已经训练好的模型，特征的名字
feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','葡萄酒','脯氨酸']

# filled 表示是否填充颜色
# rounded 表示节点的框的形状
dot_data = tree.export_graphviz(clf,feature_names= feature_name,class_names=["清酒","雪梨","贝尔摩德"]
                                ,filled=True
                                ,rounded=True
                                )

graph = graphviz.Source(dot_data)
print(graph)