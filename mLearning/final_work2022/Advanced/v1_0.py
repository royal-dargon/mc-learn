# 这个版本目前的思路是采用word2vec进行编码的手段
import jieba.posseg as jb
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier

# 这个是一个对标签进行处理的字典
label2code = {
    '正常': 0,
    '非此即彼': 1,
    '以偏概全': 2,
    '心理过滤': 3,
    '否定正面思想': 4,
    '读心术': 5,
    '先知错误': 6,
    '放大': 7,
    '缩小': 8,
    '情绪化推理': 9,
    '应该句式': 10,
    '乱贴标签': 11,
    '罪责归己': 12,
    '罪责归他': 13,
    '哀伤与丧失': 14,
    '角色间冲突': 15,
    '角色内冲突': 16,
    '角色演变': 17,
    '人际缺陷': 18
}


# 用来获取停词表的信息
def get_stop_word(file_name):
    with open(file_name, encoding='utf-8') as f:
        stop_word_list = [word.strip('\n') for word in f.readlines()]
    # print(stop_word_list)
    return stop_word_list


# 这个函数希望去实现的是获取csv文件中的数据
def get_data():
    # 这里由于Windows上的一些问题，我便没有选择将这个路径写成变量的形式
    d = pd.read_csv(r'C:\Users\hp\PycharmProjects\mc-learn\mLearning\final_work2022\data.csv', encoding='utf-8')
    df = np.array(d)
    data = df[1:, 13:]
    x_y = []
    for f in data:
        # print(f)
        t = []
        if f[-1] == "" or f[-1] == " ":
            continue
        for i in range(18):
            if f[i] == 1:
                break
        label = i + 1
        if i == 17 and f[i] == 0:
            continue
        t.append(f[-1])
        t.append(label)
        x_y.append(t)

    # 用来对数据标签的分布进行统计
    count = [0 for i in range(18)]
    for y in x_y:
        count[y[-1]-1] = count[y[-1]-1] + 1
    return np.array(x_y), count


# 一个起到增强作用的函数，希望能够平衡数据各个标签，这个函数是考虑到当前数据集的特殊性被迫采用的手段
def strength_label(data, count):
    t = []
    for text in data:
        if text[-1] != 18 and count[int(text[-1])-1] <= 10:
            for i in range(1000):
                t.append(text)
        elif text[-1] != 18 and count[int(text[-1])-1] <= 100:
            for i in range(60):
                t.append(text)
        elif text[-1] != 18 and count[int(text[-1])-1] <= 1000:
            for i in range(15):
                t.append(text)
        elif text[-1] != 18 and count[int(text[-1])-1] <= 3000:
            for i in range(8):
                t.append(text)
    t = np.array(t)
    data = np.vstack((data, t))
    count = [0 for i in range(18)]
    for y in data:
        count[int(y[-1]) - 1] = count[int(y[-1]) - 1] + 1
    print(count)
    return data


# 这里是对数据进行处理的函数，主要起到的作用是将数据进行随机的打乱操作
def random_word(data, label):
    seed_1 = 1
    random_order = list(range(len(data)))
    np.random.seed(seed_1)
    np.random.shuffle(random_order)
    data = [data[i] for i in random_order]
    label = [label[i] for i in random_order]
    return data, label


# 分词的函数
def get_text(data):
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')
    stop_words = get_stop_word(r'C:\Users\hp\PycharmProjects\mc-learn\mLearning\final_work2022\stopwords.txt')
    word_lists = []
    for text in data:
        words = [w.word for w in jb.cut(str(text)) if w.word not in stop_words and w.flag in flags]
        word_lists.append(words)
    return word_lists


# 获取特征词的词表，这里我们设置成一万维
def text_features(data):
    word_dict = []
    n = 0
    for words in data:
        if n >= 1000:
            break
        for word in words:
            if word not in word_dict and len(word) < 5:
                word_dict.append(word)
                n += 1
    return word_dict


# 将出入后的数据变为词向量的形式
def my_word2vec(data, feature_word):
    def actions(text, feature):
        feature_words = set(feature)
        feature = [1 if word in text else 0 for word in feature_words]
        return feature
    features = [actions(words, feature_word) for words in data]
    pca = PCA(n_components=3)
    features = pca.fit_transform(features)
    return features


# 一个使用word2vec生成词向量的函数
def use_word2vec(sentences):
    v_size = 100
    window = 5
    min_count = 1
    model = Word2Vec(sentences, vector_size=v_size, window=window, min_count=min_count)
    # 生成一个这样的文件希望下一次还是可以继续去使用
    model.save("word2vec.model")
    return model


# 这是一个通过word2vec生成的model进行句向量的生成的函数
def create_vec(data, model):
    w = model.wv.index_to_key
    print(w)
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
    print(n)
    for i in range(n):
        if len(data[i]) > 1:
            res_label.append(label[i])
            res_data.append(data[i])
    return res_data, res_label


# 下面开始对分类器的选择进行编写
def classify_bayes(train_data, train_label, test_data, test_label):
    classify = BernoulliNB()
    classify.fit(train_data, train_label)
    print(classify.predict(test_data))
    test_accuracy = classify.score(test_data, test_label)
    return classify, test_accuracy


# 这是我希望测试的第二个分类器选择的模型
def classify_decision_tree(train_data, train_label, test_data, test_label):
    classify = tree.DecisionTreeClassifier(random_state=30, splitter="random", max_depth=5)
    classify.fit(train_data, train_label)
    print(classify.predict(test_data))
    test_accuracy = classify.score(test_data, test_label)
    return classify, test_accuracy


# 这是我希望进行测试的第三个分类器方式
def classify_network(train_data, train_label, test_data, test_label):
    classify = MLPClassifier(solver='lbfgs', alpha=1e-5,
                             hidden_layer_sizes=(5, 2), random_state=1)
    classify.fit(train_data, train_label)
    acc = classify.score(test_data, test_label)
    return classify, acc


def main():
    data, count = get_data()
    # data = strength_label(data, count)
    data0 = data[:, 0]
    label0 = data[:, 1]
    data3, label3 = random_word(data0, label0)
    train_data = data3[:int(0.75*len(data))]
    train_label = label3[:int(0.75*len(data))]
    data2 = data3[int(0.75*len(data)):]
    label2 = label3[int(0.75*len(data)):]
    data0 = get_text(data0)
    model = use_word2vec(data0)     # 这个是去生成词向量的函数
    # for word in model.wv.index_to_key():
    #     print(word, model.wv[word])
    # train_data, train_label = random_word(data1, label1)
    # 这个是之前使用特征词提取手段的方式
    # word_dict = text_features(train_data)
    train_data, train_label = fix_empty(get_text(train_data), train_label)
    test_data, test_label = fix_empty(get_text(data2), label2)
    # print(test_data)
    # train_data = my_word2vec(train_data, word_dict)
    train_data = create_vec(train_data, model)
    test_data = create_vec(test_data, model)
    # test_data = my_word2vec(get_text(data2), word_dict)
    # print(train_data)
    # print(test_data)
    model_1, acc = classify_network(train_data, train_label, test_data, test_label)
    print("the accuracy is " + str(acc))
    print(model_1.predict(train_data[:1000]), train_label[:1000])


if __name__ == "__main__":
    main()