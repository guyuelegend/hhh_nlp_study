# coding: utf-8
"""
前言：
    1. 建立神经网络的计算模型：搭积木:你要懂需要哪些网络层，每层使用是为什么，有什么好处与坏处
    2. 样本数据处理与数据构建
    3. 评价测试样本==>测试集
    4. 使用模型预测
    5. 模型训练流程
增加：
    1. 词典
    2. 字符序列化
"""
import random
import time
import sys

import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import json
import os

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, 'Dataset')
if not os.path.isdir(data_path):
    os.mkdir(data_path)

# 模型权重路径
model_path = os.path.join(data_path, "diy_model.bin")
# 字符字典路径
dict_path = os.path.join(data_path, "dict.json")
# 损失迭代曲线路径
loss_path = os.path.join(data_path, "loss_curve.jpg")


class TorchModel(nn.Module):
    """
    1. 初始化实例化对象名称空间
    2. 前向计算的计算过程
    """

    def __init__(self, dict_len, str_dim, hidden_size, output_size):
        """
        1. Embedding层的定义：需要总字典的大小给定范围，每个字符所对应的向量维度
        2. RNN层的定义：rnn中h层的维度，单个输入向量的维度
        3. Liner层的定义：主要是传到输出层，并利用softmax对输出归一化
        4. loss损失函数的定义：因为判断字符所在字符串中的位置，可以看做是一个多分类问题，所以采用交叉熵函数
        :param dict_len: 字符字典的总长度
        :param str_dim: 每个字符向量的维度
        :param hidden_size: rnn隐藏层的维度
        :param output_size: 模型输出，分类类别个数
        """
        super().__init__()
        self.layer1 = nn.Embedding(dict_len, str_dim, padding_idx=0)
        self.layer2 = nn.RNN(str_dim, hidden_size, bias=False, batch_first=True)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.activate = torch.softmax
        self.loss_func = nn.functional.cross_entropy

    def forward(self, x, y=None):
        """
        1. Embedding层的计算：样本个数*字符串长度==>样本个数*(字符串长度*单字符维度)
        2. RNN层的计算：样本个数*(字符串长度*单字符维度) ==>(样本个数*字符串长度*rnn隐藏层的维度,1*rnn隐藏层的维度)
        3. Liner层计算：1*rnn隐藏层的维度 ==> 1*分类类别个数
        4. 根据输入样本的真实类别进行损失函数的计算。
        :param x: 1*字符串长度
        :param y: 样本分类真实值
        :return: 根据样本分类y分别返回损失值与预测值
        """
        x = self.layer1(x)
        rnn_out, hidden = self.layer2(x)
        x = rnn_out[:, -1, :]
        x = self.layer3(x)
        y_pred = self.activate(x, dim=1)
        if y is not None:
            return self.loss_func(y_pred, y)
        else:
            return y_pred


def build_str_set():
    """
    构建字典，以A-F作为字符字典，我们判断一个字符的位置，所以我们可以将A-F打乱顺序来训练字符的位置
    以json文件形式保存字符字典，方便数据存取
    """
    str_dic = {"pad": 0}
    lst = [chr(i) for i in range(65, 65 + 6)]
    for index, value in enumerate(lst):
        str_dic.update({value: index + 1})
    str_dic["unk"] = len(str_dic)
    with open(dict_path, mode='wt', encoding='utf-8') as f:
        json.dump(str_dic, f)
    return str_dic


def build_data(num):
    """
    以A-F的字符集，打乱顺序,以A字符中字符串中的位置贴标签
    注意对于方差的损失函数，预测值就算y只是一个标量，也要写成[y]作为输出的维度
    但是交叉熵的损失函数，预测值是采用one-hot形式，将降为一维，y不用加[]
    """
    random.seed(int(time.time()))
    X = []
    Y = []
    tmp_str_to_int = [i for i in range(1, 7)]
    for k in range(num):
        random.shuffle(tmp_str_to_int)
        tmp = tmp_str_to_int[:]
        X.append(tmp)
        Y.append(tmp.index(1))
    return torch.LongTensor(X), torch.LongTensor(Y)


def evaluate(model):
    x, y = build_data(100)
    x, y = x.tolist(), y.tolist()
    model.eval()
    with torch.no_grad():
        y_pre = model(torch.LongTensor(x))
        y_pre = y_pre.tolist()
        correct, wrong = 0, 0
        for index, value in enumerate(y_pre):
            y_class = value.index(max(value))
            if y[index] == y_class:
                correct += 1
            else:
                wrong += 1
        print("The correct rate of the model is {:.2f}% ".format(float(correct / (correct + wrong) * 100)))
        return correct / (correct + wrong)


def predicate():
    str_dim = 4
    hidden_size = 10
    output_size = 6
    sample_size = 20
    with open(dict_path, mode='rt', encoding='utf-8') as f:
        vocab = json.load(f)
    model = TorchModel(len(vocab), str_dim, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    x, y = build_data(sample_size)
    with torch.no_grad():
        y_pre = model(x)
        x = x.numpy()
        y_pre = y_pre.tolist()
        for i in range(len(x)):
            y_res = y_pre[i].index(max(y_pre[i]))
            print("字符串：{}， 网络模型预测的类别为：{}".format(x[i].round(decimals=4), y_res))


def main_run():
    """
    1. 参数配置，置前好调整
    2. 字符字典的生成，保存字符字典
    3. 网络模型生成
    4. 优化器选择
    5. 构建训练数据
    """
    str_dim = 4
    hidden_size = 10
    output_size = 6
    learning_rate = 0.005
    iter_time = 1000
    batch_size = 20
    sample_size = 400

    vocab = build_str_set()

    model = TorchModel(len(vocab), str_dim, hidden_size, output_size)

    optimize = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_train, y_train = build_data(sample_size)

    log = []
    for epoch in range(iter_time):
        model.train()
        loss_record = []
        for i in range(sample_size // batch_size):
            x = x_train[i * batch_size:(i + 1) * batch_size]
            y = y_train[i * batch_size:(i + 1) * batch_size]
            loss = model(x, y)

            loss.backward()

            optimize.step()

            optimize.zero_grad()

            loss_record.append(loss.item())
        accuracy = evaluate(model)
        loss_ave = sum(loss_record) / len(loss_record)
        print("The loss value of the {} generation model: {:.2f}".format(epoch + 1, loss_ave))
        log.append((accuracy, loss_ave))
    torch.save(model.state_dict(), model_path)

    plt.rcParams["font.family"] = "FangSong"
    plt.plot(range(len(log)), [i[0] for i in log], color='r', label="accuracy_rate")
    plt.plot(range(len(log)), [i[1] for i in log], color='g', label="loss_value")
    plt.xlabel("iter time")
    plt.legend()
    plt.savefig(loss_path, bbox_inches='tight', pad_inches=0)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


if __name__ == '__main__':
    main_run()
    predicate()
