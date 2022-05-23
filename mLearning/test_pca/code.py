# 这个文档是希望能对pca的操作起到一个测试与学习的作用
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# 导入了波斯顿房价的数据
house_price = datasets.load_boston()
# keys = house_price.keys()
x = house_price.data
x_std = StandardScaler().fit_transform(x)
# print(x_std)
pca = PCA(n_components=5)
new_x = pca.fit_transform(x_std)
print(new_x)
