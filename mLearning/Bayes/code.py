from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB

iris = datasets.load_iris()
gnb = GaussianNB()
mnb = MultinomialNB()
y_pred = mnb.fit(iris.data, iris.target).predict(iris.data)
# print(y_pred)


print("Number of mislabeled points out of a total %d points : %d"% (iris.data.shape[0],(iris.target != y_pred).sum()))
