# -*-coding:utf-8-*-
import torch
from torch import nn
from torch.optim import SGD, Adam


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = Encoder(config)
        self.loss = nn.CosineEmbeddingLoss()

    # def forward(self, s1, s2=None, target=None):
    #     """
    #     1. 当传入单个句子的时候，只对句子进行向量化
    #     2. 当传入两个句子的时候，对两个文本进行匹配：
    #         a. 当没有正确标签的时候，我们采用余弦距离计算
    #         b. 当有正确标签的时候，我们采用loss公式计算
    #     tip: 模型输出batch_size*hidden_size，最后将每个hidden_size向量输出来计算损失。
    #     """
    #     if s2 is not None:
    #         vector1 = self.sentence_encoder(s1)
    #         vector2 = self.sentence_encoder(s2)
    #         if target is not None:
    #             return self.loss(vector1, vector2, target.squeeze())
    #         else:
    #             return self.cosine_distance(vector1, vector2)
    #     else:
    #         return self.sentence_encoder(s1)
    def forward(self, s1, s2=None, s3=None):
        """
        1. 当传入单个句子的时候，只对句子进行向量化
        2. 当传入两个句子的时候，对两个文本进行匹配：
            a. 当没有正确标签的时候，我们采用余弦距离计算
            b. 当有正确标签的时候，我们采用loss公式计算
        tip: 模型输出batch_size*hidden_size，最后将每个hidden_size向量输出来计算损失。
        """
        if s2 is not None and s3 is not None:
            vector1 = self.sentence_encoder(s1)
            vector2 = self.sentence_encoder(s2)
            vector3 = self.sentence_encoder(s3)
            return self.cosine_triplet_loss(vector1, vector2, vector3)
        else:
            return self.sentence_encoder(s1)

    @staticmethod
    def cosine_distance(vec1, vec2):
        """
        余弦距离cosθ = a * b / (|a|*|b|)
        而正则化计算是a_normal = a / |a|
        计算两个向量的余弦值就等于两个向量正则化相乘，然后相加；
        因为两向量的叉集本身是向量对位相乘，然后求和。
        """
        vector1 = torch.nn.functional.normalize(vec1, dim=-1)
        vector2 = torch.nn.functional.normalize(vec2, dim=-1)
        # return torch.cosine_similarity(vec1, vec2, dim=-1, eps=1e-8)
        return 1 - torch.sum(torch.mul(vector1, vector2))

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = 1 - self.cosine_distance(a, p)
        an = 1 - self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
            print(diff[diff.gt(0)])
        return torch.mean(diff[diff.gt(0)])
        # torch.gt(0)是取greater大于0的数据,返回torch（True，False，False...）
        # torchdata[torch（True，False，False...）]可以选择索引的数据


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        vocab_size = config["vocab_size"] + 1
        hidden_size = config["hidden_size"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.Linear(hidden_size, hidden_size)
        # bidirectional参数会水平拼接，会将hidden_size放大两倍
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # batch_size*seq_size*hidden_size
        x = self.embedding(x)
        # batch_size*seq_size*hidden_size
        x = self.layer(x)
        if isinstance(x, tuple):
            x = x[0]
        x = x.transpose(1, 2)
        # batch_size*hidden_size
        x = nn.functional.avg_pool1d(x, kernel_size=x.shape[-1]).squeeze()
        return x


def choose_optimizer(config, model):
    optim_type = config["optimizer_type"]
    learn_rate = config["learn_rate"]
    if optim_type.upper() == "SDG":
        return SGD(model.parameters(), lr=learn_rate)
    else:
        return Adam(model.parameters(), lr=learn_rate)


if __name__ == '__main__':
    from core.data_loader import data_load
    from conf.setting import Config

    dg = data_load(Config, Config["valid_data_path"]).__iter__()
    input_id, label = next(dg)
    print(input_id)
    # print("\n\n")
    # print(label.squeeze())
    model = Encoder(Config)
    model.train()
    with torch.no_grad():
        y_predicate = model(input_id)
        print(len(y_predicate[0]))
        # for pre,lab in zip(y_predicate,label):
        #     print(pre,lab)
        # print(next(zip(y_predicate,label)))
