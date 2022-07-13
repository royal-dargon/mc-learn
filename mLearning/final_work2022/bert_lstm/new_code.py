import pandas as pd
from transformers import BertTokenizer, BertModel
import numpy as np
import os
import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
from torch.utils.data import TensorDataset, DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

n_categories = 15
all_categories = ['民生', '文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '军事', '旅游', '国际', '证券', '农业', '电竞']

test_dates = pd.read_csv('data/test.txt', names=['content', 'kind'], header=None, sep="_!_")
train_dates = pd.read_csv('data/train.txt', names=['content', 'kind'], header=None, sep="_!_")
val_dates = pd.read_csv('data/val.txt', names=['content', 'kind'], header=None, sep="_!_")

train_dates = np.array(train_dates)
train_sent, train_label = train_dates[:, 0], train_dates[:, 1]
# test_dates = np.array(test_dates)
# test_sent, test_label = test_dates[:, 0], test_dates[:, 1]
# val_dates = np.array(val_dates)
# val_sent, val_label = val_dates[:, 0], val_dates[:, 1]


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


tz = BertTokenizer.from_pretrained("bert_cn")
bert_model = BertModel.from_pretrained("bert_cn")
n_hidden = 128
# model = LSTM(768, n_hidden, n_categories)
model = torch.load('rfc.pkl')
criterion = nn.NLLLoss()


# 这个函数希望实现的效果是将句子生成句向量
def sent_tensor(sent):
    tokens = tz(sent, padding=True, max_length=100, truncation=True, return_tensors='pt')
    # input_ids = torch.tensor(tokens['input_ids'])
    output = bert_model(**tokens)
    # 这个大小是1 * 句子的长度 * 768
    last_hidden_state = output[0]
    return last_hidden_state


# 这个函数的作用是希望能够将标签的值变成tensor
def category_tensor(category):
    category_id = torch.tensor([all_categories.index(category)], dtype=torch.long)
    return category_id


def category_output(out):
    top_n, top_i = out.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


learning_rate = 0.005


# target_line_tensor
def train(input_line_tensor, target_line_tensor):
    hidden = model.init_hidden()
    cn = model.init_cn()
    model.zero_grad()
    for k in range(input_line_tensor.size()[1]):
        input = input_line_tensor[0][k].view(1, -1)
        output, (hidden, cn) = model(input, hidden, cn)
    loss = criterion(output, target_line_tensor)
    loss.backward(retain_graph=True)

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


# test = sent_tensor(test_sent[0], tz, bert_model)
# output, loss = train(test, category_tensor(all_categories[test_label[0]]))
# a, b = category_output(output)

print_every = 50
plot_every = 10

current_loss = 0
all_losses = []


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for i in range(1000):
    line_tensor = sent_tensor(train_sent[i])
    category_id = category_tensor(all_categories[train_label[i]])
    output, loss = train(line_tensor, category_id)
    current_loss += loss

    if (i+1) % print_every == 0:
        guess, guess_i = category_output(output)
        correct = '✓' if guess_i == train_label[i] else '✗ (%s)' % all_categories[train_label[i]]
        print('%d %d%% (%s) %.4f %s / %s %s' % (i+1, (i+1) / 1000 * 100, time_since(start), loss, train_sent[i], guess, correct))

    if (i+1) % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(model, "rfc1.pkl")

plt.figure()
plt.plot(all_losses)
plt.show()


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


# predict(test_sent[1])
# predict(test_sent[2])
# predict(test_sent[3])



