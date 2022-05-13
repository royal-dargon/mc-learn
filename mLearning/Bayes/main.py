import jieba
import pandas as pd
import numpy as np
import sklearn.preprocessing as sp

data = pd.read_csv('result1.csv', encoding='utf-8')
df = np.array(data)
X = df[:, 4]
Y = df[:, 0]

lbe = sp.LabelEncoder()
y_lable = lbe.fit_transform(Y)


# 读取停词表中的信息
def get_stop_list(file_name):
    with open(file_name, encoding='utf-8') as f:
        stop_word_list = [word.strip('\n') for word in f.readlines()]
    # print(stop_word_list)
    return stop_word_list


# 这是一个使用结巴分词进行出入输入语句的函数
def data_process(x):
    x_list = []
    stop_list = get_stop_list("stopwords.txt")
    for word in x:
        word = str(word)
        res = ''
        str_list = jieba.cut(word, use_paddle=True)
        for s in str_list:
            if s not in stop_list:
                res += s
        x_list.append(res)
    return x_list


# 这是通过处理后得到的x
x_lists = data_process(X)
x_list2 = []
for x_list in x_lists:
    l = jieba.lcut(str(x_list))
    x_list2.append(l)


# 构建词向量



