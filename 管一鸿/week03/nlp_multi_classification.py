# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(
            input_size=vector_dim,
            hidden_size=vector_dim,
            num_layers=1,
            batch_first=True,
        )  # RNN层
        self.classify = nn.Linear(vector_dim, sentence_length + 1)  # 分类头，用于多分类任务
        self.activation = nn.Softmax(dim=1)  # Softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # 损失函数采用交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim)
        x = rnn_out[:, -1, :]  # 取最后一个时间步的隐藏状态 (batch_size, vector_dim)
        y_pred = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, sentence_length + 1)
        # y_pred = self.activation(x)  # (batch_size, sentence_length + 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本

# def build_sample(vocab, sentence_length):
#     # 随机采样，避免字符重复出现
#     x = random.sample(list(vocab.keys()), sentence_length)  # 使用random.sample保证不重复
#     # 指定哪些字出现时为正样本
#     if "我" in x:
#         y = x.index("我")
#     # 指定字都未出现，则为负样本
#     else:
#         y = sentence_length
#     x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
#     return x, y

# 上面的build方法，会导致“其他”（也就是没有“我”在文本中的这个类别)的数据量过大，使得模型预测效果很差
# 下面直接放弃了“其他”类别
def build_sample(vocab, sentence_length):
    # 确保每个样本一定包含"我"
    x = [random.choice(list(vocab.keys() - "我")) for _ in range(sentence_length - 1)]  # 从其它字中随机选
    # 在随机位置插入"我"
    my_position = random.randint(0, sentence_length - 1)  # 随机选一个位置放"我"
    x.insert(my_position, "我")

    # 标签y是"我"在句子中的位置
    y = my_position

    # 将字转换成序号
    x = [vocab.get(word, vocab['unk']) for word in x]  # 使用字典将字转换为索引

    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        probabilities_classes = torch.argmax(y_pred, dim=1)
        correct = (probabilities_classes == y).sum().item()
        wrong = 200 - correct  # 计算错误的数量
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 70  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.001  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
        pred_labels = torch.argmax(result, dim=1)
        probability_labels = torch.softmax(result, dim=1)
    for vec, res, prob in zip(x, pred_labels, probability_labels):
        prob_value = prob[res].item()  # 获取该类别的概率值
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, res.item(), prob_value))  # 打印结果


if __name__ == "__main__":
    main()
    # test_strings = ["n我l", "z我d", "rd我", "我lw"]
    test_strings = ["fnk我ey", "wzv我fv", "rq我wdh", "nkwwl我"]
    predict("model.pth", "vocab.json", test_strings)
