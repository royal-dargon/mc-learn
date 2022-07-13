# 这个是通过word2vec来实现编码的手段
import jieba.posseg as jb
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from gensim.models import Word2Vec
import joblib


def get_stop_words(file_name):
    with open(file_name, encoding='utf-8') as f:
        stop_word_list = [word.strip('\n') for word in f.readlines()]
    return stop_word_list


# 一个使用word2vec生成词向量的函数
def use_word2vec(sentences):
    v_size = 100
    window = 5
    min_count = 1
    model = Word2Vec(sentences, vector_size=v_size, window=window, min_count=min_count)
    # 生成一个这样的文件希望下一次还是可以继续去使用
    model.save("word2vec.model")
    return model


# 分词的函数
def get_text(data):
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')
    stop_words = get_stop_words(r'C:\Users\hp\PycharmProjects\mc-learn\mLearning\final_work2022\stopwords.txt')
    word_lists = []
    for text in data:
        words = [w.word for w in jb.cut(str(text)) if w.word not in stop_words and w.flag in flags]
        word_lists.append(words)
    return word_lists


# 这是一个通过word2vec生成的model进行句向量的生成的函数
def create_vec(data, model):
    w = model.wv.index_to_key
    def actions(words):
        n = len(words)
        feature = np.zeros(100)
        for i in range(len(words)):
            if words[i] == 'NaN':
                n = n - 1
                continue
            if words[i] in w:
                feature = feature + model.wv[words[i]]
            else:
                n = n - 1
                continue
        feature = feature / n
        return feature
    features = [actions(words) for words in data]
    return features


# 这个函数需要解决的问题是假如在分词后出现了空的情况
def fix_empty(data, label):
    n = len(data)
    res_data = []
    res_label = []
    for i in range(n):
        if len(data[i]) > 1:
            res_label.append(label[i])
            res_data.append(data[i])
    return res_data, res_label


# 这是我希望进行测试的第三个分类器方式
def classify_network(train_data, train_label, test_data, test_label):
    classify = BernoulliNB()
    # classify = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                           hidden_layer_sizes=(5, 2), random_state=1)
    print(len(train_label))
    classify.fit(train_data, train_label)
    acc = classify.score(test_data, test_label)
    return classify, acc


def fix_nan(data, label):
    res_data = []
    res_label = []
    for i in range(len(data)):
        if np.isfinite(data[i]).all() == True:
            res_data.append(data[i])
            res_label.append(label[i])

    return np.array(res_data), np.array(res_label)



def main():
    test_dates = pd.read_csv('data/test.txt', names=['content', 'kind'], header=None, sep="_!_")
    train_dates = pd.read_csv('data/train.txt', names=['content', 'kind'], header=None, sep="_!_")
    # val_dates = pd.read_csv('data/val.txt', names=['content', 'kind'], header=None, sep="_!_")

    train_dates = np.array(train_dates)
    train_sent, train_label = train_dates[:, 0], train_dates[:, 1]
    test_dates = np.array((test_dates))
    test_sent, test_label = test_dates[:, 0], test_dates[:, 1]
    # model = use_word2vec(train_sent)     # 这个是去生成词向量的函数
    model = joblib.load('word2vec.model')
    train_data, train_label = fix_empty(get_text(train_sent[:15000]), train_label[:15000])
    test_data, test_label = fix_empty(get_text(test_sent[:3000]), test_label[:3000])
    train_data = create_vec(train_data[:20000], model)
    test_data = create_vec(test_data[:2000], model)
    train_data, train_label = fix_nan(train_data, train_label[:20000])
    test_data, test_label = fix_nan(test_data, test_label[:2000])
    model_1, acc = classify_network(train_data, train_label, test_data, test_label)
    print("the accuracy is " + str(acc))
    print(model_1.predict(train_data[:100]), train_label[:100])


if __name__ == "__main__":
    main()
