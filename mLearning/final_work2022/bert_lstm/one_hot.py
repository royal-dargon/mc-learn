# 这个是采用one-hot编码方式
import jieba.posseg as jb
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn import tree

n_categories = 15
all_categories = ['民生', '文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '军事', '旅游', '国际', '证券', '农业', '电竞']


def get_stop_words(file_name):
    with open(file_name, encoding='utf-8') as f:
        stop_word_list = [word.strip('\n') for word in f.readlines()]
    return stop_word_list


# 分词的函数
def get_text(data):
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')
    stop_words = get_stop_words(r'C:\Users\hp\PycharmProjects\mc-learn\mLearning\final_work2022\stopwords.txt')
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
        if n >= 2000:
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
    return features


# 下面开始对分类器的选择进行编写
def classify_bayes(train_data, train_label, test_data, test_label):
    classify = MultinomialNB()
    classify.fit(train_data, train_label)
    # print(classify.predict(test_data))
    test_accuracy = classify.score(test_data, test_label)
    return classify, test_accuracy

# 这是我希望测试的第二个分类器选择的模型
def classify_decision_tree(train_data, train_label, test_data, test_label):
    classify = tree.DecisionTreeClassifier(random_state=30, splitter="random", max_depth=5)
    classify.fit(train_data, train_label)
    print(classify.predict(test_data))
    test_accuracy = classify.score(test_data, test_label)
    return classify, test_accuracy


def process_datas(datas):
    res = []
    for d in datas:
        for w in d:
            res.append(d)
    return res


def word_picture(datas):
    # print(datas)

    new_datas = process_datas(datas)
    space_list = ' '.join(new_datas)  # 空格链接词语
    wc = WordCloud(width=1400, height=2200,
                   background_color='white',
                   mode='RGB',
                   max_words=500,
                   max_font_size=150,
                   relative_scaling=0.6,  # 设置字体大小与词频的关联程度为0.4
                   random_state=50,
                   scale=2
                   ).generate(space_list)
    plt.imshow(wc)  # 显示词云
    plt.axis('off')  # 关闭x,y轴
    plt.show()  # 显示
    wc.to_file('test1_ciyun.jpg')  # 保存词云图

def main():
    test_dates = pd.read_csv('data/test.txt', names=['content', 'kind'], header=None, sep="_!_")
    train_dates = pd.read_csv('data/train.txt', names=['content', 'kind'], header=None, sep="_!_")
    # val_dates = pd.read_csv('data/val.txt', names=['content', 'kind'], header=None, sep="_!_")

    train_dates = np.array(train_dates)
    train_sent, train_label = train_dates[:, 0], train_dates[:, 1]
    test_dates = np.array((test_dates))
    test_sent, test_label = test_dates[:, 0], test_dates[:, 1]
    datas = get_text(train_sent[:20000])
    word_dict = text_features(datas)
    train_data = my_word2vec(datas, word_dict)
    test_data = my_word2vec(get_text(test_sent[:500]), word_dict)
    train_label = train_label[:20000].astype('int')
    test_label = test_label[:500].astype('int')
    model_1, acc = classify_decision_tree(train_data, train_label, test_data, test_label)
    print("the accuracy is " + str(acc))
    print(model_1.predict(train_data[:10]), train_label[:10])


if __name__ == "__main__":
    main()

