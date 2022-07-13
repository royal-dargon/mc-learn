import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import numpy as np
import pandas as pd


class LSTM(nn.Module):
    # 这里主要是三个参数，分别是输入的size，隐藏层层层数，输出的size
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        # 这里希望对bert编码产生的结果进行一个降维的操作
        self.x2x = nn.Linear(input_size, 32)
        # 这里有三个参数分别是输入，隐藏层的数量，层数
        self.lstm = nn.LSTM(32, hidden_size, 1)
        self.o2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_code, hidden_code, context):
        id_code = self.x2x(input_code)
        # out的是一个三维张量：句子中字的数量，批量大小，LSTM方向数量*隐藏向量维度
        # hn维度(num_layers * num_directions, batch, hidden_size)
        out, (hn, cn) = self.lstm(id_code, (hidden_code, context))
        out = self.o2o(out)
        out = self.dropout(out)
        out = self.softmax(out)
        return out, (hn, cn)

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def init_cn(self):
        return torch.zeros(1, self.hidden_size)


all_categories = ['民生', '文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '军事', '旅游', '国际', '证券', '农业', '电竞']
model = torch.load('rfc1.pkl')


tz = BertTokenizer.from_pretrained("bert_cn")
bert_model = BertModel.from_pretrained("bert_cn")


# 这个函数希望实现的效果是将句子生成句向量
def sent_tensor(sent):
    tokens = tz(sent, padding=True, max_length=100, truncation=True, return_tensors='pt')
    # input_ids = torch.tensor(tokens['input_ids'])
    output = bert_model(**tokens)
    # 这个大小是1 * 句子的长度 * 768
    last_hidden_state = output[0]
    return last_hidden_state


# 下面是一个模型的评估部分
def evaluate(sent):
    hidden = model.init_hidden()
    cn = model.init_cn()

    for i in range(sent.size()[0]):
        sent = sent[0][i].view(1, -1)
        output, (hidden, cn) = model(sent, hidden, cn)

    return output


# 下面是一个模型的预测的实现部分
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(sent_tensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


# 这个是对测试集的准确度进行验证的部分
test_dates = pd.read_csv('data/test.txt', names=['content', 'kind'], header=None, sep="_!_")
test_dates = np.array(test_dates)
test_sent, test_label = test_dates[:, 0], test_dates[:, 1]


def predict_test():
    count = 0
    for i in range(500):
        output = evaluate(sent_tensor(test_sent[i]))
        topv, topi = output.topk(1, 1, True)
        category_index = topi[0][0].item()
        if category_index == test_label[i]:
            count = count + 1
        # print(test_sent[i], all_categories[category_index], all_categories[test_label[i]])
    print('the accuracy is %s%%' % (count / 500 * 100))


# predict("巅峰时期的梅西是什么水平？")
# predict("巅峰时期的清华是什么水平？")
# predict("巅峰时期的玉米产业是什么水平？")
#
# # predict_test()
# predict("凯尔特人G3击败勇士")
# # predict("李劲哲")
predict("ANY anti Trump propaganda from Gaga and my TV goes off immediately and I will never watch the NFL again. They hired her and are responsible for any unsavory political speech she may spew.")

