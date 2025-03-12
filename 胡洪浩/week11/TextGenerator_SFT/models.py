#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：TextGenerator_CausalLanguageModel 
@File    ：models.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/3/4 17:28 
"""
import time
import torch
from torch import nn
from transformers import BertModel
from torch.optim import SGD, Adam


class CausalLanguageModel(nn.Module):
    def __init__(self, config):
        super(CausalLanguageModel, self).__init__()
        self.model_type = config["model_type"]
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if config["model_type"] == "bert":
            self.encoder = BertModel.from_pretrained(config["bert_model_path"], return_dict=False,
                                                     attn_implementation='eager')
            hidden_size = self.encoder.config.hidden_size
            vocab_size = self.encoder.config.vocab_size
        else:
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, target=None,mask=None):
        if target is not None:
            # 定义一个mask的上三角为零，mask为零的元素对应在attention位置会替换成-inf负无穷
            if mask is None:
                mask = torch.tril(torch.ones(x.shape[0], x.shape[1], x.shape[1]))
            if torch.cuda.is_available():
                mask = mask.cuda()
            if self.model_type == "bert":
                x, _ = self.encoder(x, attention_mask=mask)
                # x, _ = self.encoder(x)
            else:
                x = self.embedding(x)
                x, _ = self.encoder(x)
            y = self.classify(x)
            # 300 * vocab_size   // 300*1
            # print(y.view(-1, y.shape[-1]).shape, target.view(-1).shape)
            return self.loss_func(y.view(-1, y.shape[-1]), target.view(-1))

        else:
            if self.model_type == "bert":
                x, _ = self.encoder(x)
            else:
                x = self.embedding(x)
                x, _ = self.encoder(x)
            y = torch.nn.functional.softmax(self.classify(x), dim=-1)
            return y


def choose_optim(config, demo):
    optim_type = config["optim_type"]
    learning_rate = config["learning_rate"]
    if optim_type.upper() == "SGD":
        return SGD(demo.parameters(), lr=learning_rate)
    else:
        return Adam(demo.parameters(), lr=learning_rate)


if __name__ == '__main__':
    from dataloader import generate_data
    from setting import Config

    start = time.time()
    dg = generate_data(Config, Config["train_data_path"], False)
    print("数据加载完成，耗时：{}s".format(time.time() - start))
    model = CausalLanguageModel(Config)
    print(next(dg.__iter__()))
    batch_data = next(dg.__iter__())
    print(batch_data)
    if torch.cuda.is_available():
        model = model.cuda()
        batch_data = [d.cuda() for d in batch_data]
    batch_x, batch_y = batch_data
    loss = model(batch_x, batch_y)
    print("第一次loss: ", loss.item())
