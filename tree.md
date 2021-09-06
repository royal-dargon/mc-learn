### 决策树

* 关键概念：节点
  * 根节点：没有进边，有出边。包含最初的，针对特征的提问。
  * 中间节点：既有进边也有出边。
  * 叶子节点：有进边，没有出边，没个叶子节点都是一个类的标签

* 决策树算法的核心：
  * 如何从数据表中找出最佳节点和最佳分支
  * 如何让决策树停止生长，防止过拟合

#### sklearn中的决策树

* 模块sklearn.tree中

  里面总共包含五个类别

  下面是一个建模的示范

  ```python
  from sklearn import tree   
  
  clf = tree.DecisionTreeClassifier() #实例化
  clf = clf.fit(x_train,y_train)		#用训练集数据训练模型
  result = clf.score(x_test,y_test)	#导入测试集，从接口中调用需要的信息
  ```

  上面是三个步骤

  1. 实例化，建立评估模型对象
  2. 通过模型接口训练模型
  3. 通过模型接口提取需要的信息