import jieba
import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import joblib
from sklearn.decomposition import PCA



# 读取停词表中的信息
def get_stop_list(file_name):
    with open(file_name, encoding='utf-8') as f:
        stop_word_list = [word.strip('\n') for word in f.readlines()]
    # print(stop_word_list)
    return stop_word_list


# 这是一个使用结巴分词进行出入输入语句的函数
def data_process(x):
    word_lists = []
    for words in x:
        res = jieba.cut(str(words), use_paddle=True)
        word_list = list(res)
        # print(word_list)
        word_lists.append(word_list)
    return word_lists


# 这个函数是对文本的特征词进行提取, 存入的参数是之前进行分词完毕的词和题词表
def word_dict(word_lists, stop_words):
    # 特征列表
    feature_dict = []
    n = 1   # 用来计录提取的维度
    for words in word_lists:
        if n > 1000:
            break
        for word in words:
            if word not in stop_words and not word.isdigit() and word not in feature_dict and len(word) > 1 and len(
                    word) < 3:
                feature_dict.append(word)
                n += 1
    return feature_dict


# 这是一个根据刚才得到的feature_dict进行向量化的操作
def text_features(x_train, x_test, feature_word):
    def actions(text, feature):
        text_words = set(text)
        feature = [1 if word in feature else 0 for word in text_words]
        return feature
    train_features = [actions(feature_word, text) for text in x_train]
    test_features = [actions(feature_word, text) for text in x_test]
    pca = PCA(n_components=2)
    train_features = pca.fit_transform(train_features)
    test_features = pca.transform(test_features)
    print(test_features)
    return train_features, test_features


def text_classifer(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classfier1 = GaussianNB()
    # pca = PCA(n_components=4)
    # pca.fit(train_feature_list)
    classfier1 = classfier1.fit(train_feature_list, train_class_list)
    # print(test_feature_list)
    # print(classfier1.predict(test_feature_list))
    # print(test_class_list)
    test_accuracy = classfier1.score(test_feature_list, test_class_list)
    return test_accuracy, classfier1


# 解决不对齐的问题
def solve_zero(text_features, test_features):
    mlen = 0
    arr_list = []
    arr_list2 = []
    for text_feature in text_features:
        if mlen < len(text_feature):
            mlen = len(text_feature)
    for text in test_features:
        if mlen < len(text):
            mlen = len(text)
    for text_feature in text_features:
        np.array(text_feature)
        if len(text_feature) < mlen:
            text_feature = np.pad(text_feature, (0, mlen - len(text_feature)), 'constant', constant_values=0)
        arr_list.append(text_feature)
    for text in test_features:
        np.array(text)
        if len(text) < mlen:
            text = np.pad(text, (0, mlen - len(text)), 'constant', constant_values=0)
        arr_list2.append(text)
    return np.array(arr_list), np.array(arr_list2)


if __name__ == "__main__":
    data = pd.read_csv('result1.csv', encoding='utf-8')
    df = np.array(data)
    X = df[:, 4]
    Y = df[:, 0]
    lbe = sp.LabelEncoder()
    y_lable = lbe.fit_transform(Y)
    # print(set(y_lable))
    temp = np.array([X, y_lable])
    temp = temp.transpose()
    # print(temp[1])
    np.random.shuffle(temp)
    X, y_lable = temp[:, 0], temp[:, 1]
    # print(X)
    x_train, y_train = X[:int(len(X) * 0.75)], y_lable[:int(len(y_lable) * 0.75)]
    x_test, y_test = X[int(len(X) * 0.75):], y_lable[int(len(y_lable) * 0.75):]
    # 这里是开始进行分词的处理
    x_train = data_process(x_train)
    x_test = data_process(x_test)
    print(x_train[:5])
    # 获取停词表中的信息
    stop_words = get_stop_list("stopwords.txt")
    feature_dict = word_dict(x_train, stop_words)
    # print(feature_dict)
    train_features, test_features = text_features(x_train, x_test, feature_dict)
    # train_features, test_features = np.array(train_features), np.array(test_features)
    train_features, test_features = solve_zero(train_features, test_features)
    # print(train_features)
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)
    test_accuracy, clf = text_classifer(train_features, test_features, y_train, y_test)
    print(test_accuracy)
    # for i in range(1000):
    #     print(test_features[2][i])

    # save model
    joblib.dump(clf, 'rfc.pkl')
    # load model3
    rfc2 = joblib.load('rfc.pkl')

