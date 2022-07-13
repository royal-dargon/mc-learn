import pandas as pd
from transformers import BertTokenizer, BertModel
import numpy as np
import os
import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader


# print(tz.convert_tokens_to_ids(tz.tokenize(sent)))


class ModelConfig():
    def __init__(self):
        # 这个是获取当前文件目录的函数
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_cn")
        self.vocab_path = os.path.join(self.pretrained_model_dir, "vocab.txt")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, "train.txt")
        self.val_file_path = os.path.join(self.dataset_dir, "val.txt")
        self.test_file_path = os.path.join(self.dataset_dir, "test.txt")
        self.model_save_dir = os.path.join(self.project_dir, "cache")
        self.is_sample_shuffle = True
        self.batch_size = 64
        self.max_sen_len = 64
        # 表示十五个标签
        self.num_labels = 15
        # 表示一共训练的轮次数
        self.epochs = 10
        self.model_val_per_epoch = 2
        # 判断是否存在已经训练好的模型，如果没有的话就创建
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)


# 这个是进行文件读取数据集的类
class LoadDate():
    def __init__(self,
                 tokenizer=None,
                 random_state=2022,
                 seps="。"):
        self.tokenizer = tokenizer
        self.seps = seps
        random.seed(random_state)

    def Load_train(self):
        tz = BertTokenizer.from_pretrained("bert_cn")
        # model = BertModel.from_pretrained("bert_cn")
        test_dates = pd.read_csv('data/test.txt', names=['content', 'kind'], header=None, sep="_!_")
        train_dates = pd.read_csv('data/train.txt', names=['content', 'kind'], header=None, sep="_!_")
        val_dates = pd.read_csv('data/val.txt', names=['content', 'kind'], header=None, sep="_!_")
        train_dates = np.array(train_dates)
        train_sent, train_label = train_dates[:, 0], train_dates[:, 1]
        train_list = self.word2token(train_sent, tz)
        test_dates = np.array(test_dates)
        test_sent, test_label = test_dates[:, 0], test_dates[:, 1]
        test_list = self.word2token(test_sent, tz)
        val_dates = np.array(val_dates)
        val_sent, val_label = val_dates[:, 0], val_dates[:, 1]
        val_list = self.word2token(val_sent, tz)
        # train_label = self.label2list(train_label)
        train_label = train_label.astype(int)  # numpy强制类型转换
        test_label = test_label.astype(int)  # numpy强制类型转换
        val_label = val_label.astype(int)  # numpy强制类型转换
        train_label = torch.from_numpy(train_label).float()
        test_label = torch.from_numpy(test_label).float()
        val_label = torch.from_numpy(val_label).float()
        # print(train_label)
        return train_list, train_label, test_list, test_label, val_list, val_label

    def word2token(self, sents, tz):
        sent_list = []
        # tokens = tz(list(sent), padding=True, max_length=100, truncation=True, return_tensors='pt')
        # input_ids = torch.tensor(tokens['input_ids'])
        for sent in sents:
            tokens = tz.encode(sent[:48])
            if len(tokens) < 48 + 2:
                tokens.extend([0] * (48 + 2 - len(tokens)))
            sent_list.append(tokens)
        return torch.tensor(sent_list)

    # 这个函数是用来将label转变为one-hot编码
    def label2list(self, labels):
        res = []
        for label in labels:
            label_emb = [1 if label == i else 0 for i in range(15)]
            res.append(label_emb)
        return res

class LSTM(nn.Module):
    # 这边是三个参数，分别是输入的大小，隐藏层的大小，输出的大小
    def __init__(self, hidden_size, output_size, n_layers, drop_prob=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        # self.input_size = input_size
        # self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # Bert bert的模型也是需要嵌入到自定义的模型中
        self.bert = BertModel.from_pretrained("bert_cn")
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_size, n_layers)
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers
        # if bidirectional:
        #     self.fc = nn.Linear(hidden_size * 2, output_size)
        # else:
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # 生成bert字向量
        x = self.bert(x)[0]  # bert 字向量



        # lstm_out
        # x = x.float()
        lstm_out, (hidden_last, cn_last) = self.lstm(x, hidden)
        # print(lstm_out.shape)   #[32,100,768]
        # print(hidden_last.shape)   #[4, 32, 384]
        # print(cn_last.shape)    #[4, 32, 384]
        hidden_last_out = hidden_last[-1]  # [32, 384]

        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        # print(out.shape)    #[32,768]
        out = self.fc(out)
        out = self.softmax(out)

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        number = 1
        # if self.bidirectional:
        #     number = 2

        hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_size).zero_().float(),
                  weight.new(self.n_layers * number, batch_size, self.hidden_size).zero_().float()
                  )

        return hidden


def train(config):
    output_size = 1
    hidden_dim = 384  # 768/2
    n_layers = 2
    bidirectional = True  # 这里为True，为双向LSTM

    net = LSTM(hidden_dim, output_size, n_layers)

    train_list, train_label, test_list, test_label, val_list, val_label = LoadDate().Load_train()
    train_data = TensorDataset(train_list, train_label)
    valid_data = TensorDataset(val_list, val_label)
    test_data = TensorDataset(test_list, test_label)
    batch_size = 50
    # print(train_data)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    print(train_loader)
    lr = 2e-5
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params
    epochs = 10
    # batch_size=50
    print_every = 15
    clip = 5  # gradient clipping
    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # print(e)
        # initialize hidden state

        h = net.init_hidden(batch_size)
        counter = 0

        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            # print(counter)
            h = tuple([each.data for each in h])
            net.zero_grad()
            output = net(inputs, h)
            # print(len(inputs[0]))
            # labels = torch.unsqueeze(labels, 0)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                net.eval()
                with torch.no_grad():
                    val_h = net.init_hidden(batch_size)
                    # print(batch_size)
                    val_losses = []
                    for inputs, labels in valid_loader:
                        val_h = tuple([each.data for each in val_h])
                        # print(1111)
                        # print(len(inputs[0]))
                        output = net(inputs, val_h)
                        val_loss = criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))


# def inference(config):
#     pass


if __name__ == "__main__":
    model_config = ModelConfig()
    train(model_config)
    # inference(model_config)

