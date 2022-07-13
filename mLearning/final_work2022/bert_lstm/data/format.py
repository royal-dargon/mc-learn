# 一个格式化处理的文件
import numpy as np
import matplotlib.pyplot as plt

plt.rc("font",family='YouYuan')

# 这个是将数据中开始的编号，转变为0至14，一共十五个标签
label_map = {'100': '0', '101': '1', '102': '2', '103': '3', '104': '4', '106': '5',
             '107': '6', '108': '7', '109': '8', '110': '9', '112': '10', '113': '11', '114': '12',
             '115': '13', '116': '14'}

all_categories = ['民生', '文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '军事', '旅游', '国际', '证券', '农业', '电竞']


# 这里希望使用一个小技巧可以控制每个标签的数量，比如我们希望训练集的数量控制在3000(每个标签的数量控制在200)，测试集的数量控制在750，每个标签50
def deal():
    data_dict1 = {}
    data_dict2 = {}
    data_dict3 = {}
    np.random.seed(2022)
    raw_data = open("toutiao_cat_data.txt", 'r', encoding='utf-8').readlines()
    num_samples = len(raw_data)
    # print(raw_data[:5])
    # 这个的目的是序号的随机处理
    idx = np.random.permutation(num_samples)
    num_train, num_val = int(0.7*num_samples), int(0.2*num_samples)
    num_test = num_samples - num_train - num_val
    # 这是是对训练集与测试集进行了划分的操作，其中val是用来判断是否过度拟合的
    train_idx, val_inx, test_idx = idx[:num_train], idx[num_train:num_train + num_val], idx[-num_test:]
    f_train = open("train.txt", 'w', encoding='utf-8')
    f_val = open("val.txt", 'w', encoding='utf-8')
    f_test = open("test.txt", 'w', encoding='utf-8')

    for i in train_idx:
        r = raw_data[i].strip('\n').split('_!_')
        label, text = label_map[r[1]], r[3]
        if label_map[r[1]] not in data_dict1:
            data_dict1[label_map[r[1]]] = 0
        # if data_dict1[label_map[r[1]]] >= 300:
        #     continue
        f_train.write(text + "_!_" + label + '\n')
        data_dict1[label_map[r[1]]] += 1
    f_train.close()
    print(data_dict1)

    for i in val_inx:
        r = raw_data[i].strip('\n').split('_!_')
        label, text = label_map[r[1]], r[3]
        if label_map[r[1]] not in data_dict3:
            data_dict3[label_map[r[1]]] = 0
        if data_dict3[label_map[r[1]]] >= 10:
            continue
        f_val.write(text + "_!_" + label + '\n')
        data_dict3[label_map[r[1]]] += 1
    f_val.close()
    print(data_dict3)

    for i in test_idx:
        r = raw_data[i].strip('\n').split('_!_')
        label, text = label_map[r[1]], r[3]
        if label_map[r[1]] not in data_dict2:
            data_dict2[label_map[r[1]]] = 0
        # if data_dict2[label_map[r[1]]] >= 100:
        #     continue
        f_test.write(text + "_!_" + label + '\n')
        data_dict2[label_map[r[1]]] += 1
    f_test.close()
    print(data_dict2)

    x = [all_categories[i] for i in range(15)]
    y = [data for data in data_dict1]
    plt.bar(x, y)
    plt.show()


if __name__ == "__main__":
    deal()
