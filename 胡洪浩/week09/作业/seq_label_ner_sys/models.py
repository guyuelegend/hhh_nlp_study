#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：SeqLabelNerSystem 
@File    ：models.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/2/25 13:49 
"""
import torch
import numpy
import random
from torch import nn
from torchcrf import CRF
from torch.optim import SGD, Adam
from transformers import BertModel


# seed = 768
# torch.manual_seed(seed)
# random.seed(seed)
# numpy.random.seed(seed)
# torch.cuda.manual_seed(seed)

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        vocab_size = config["vocab_size"] + 1
        hidden_size = config["hidden_size"]
        label_num = config["label_num"]
        num_layer = config["num_layer"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.encoder = nn.LSTM(hidden_size, hidden_size,
                               batch_first=True, num_layers=num_layer,
                               bidirectional=True)
        self.classify = nn.Linear(hidden_size * 2, label_num)
        if config["model_type"] == "bert":
            self.encoder = BertModel.from_pretrained(config["bert_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
            self.classify = nn.Linear(hidden_size, label_num)
        self.crf_layer = CRF(label_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.config = config

    def forward(self, x, target=None):
        if not self.config["model_type"] == "bert":
            # batch_size*seq_size*hidden_size
            x = self.embedding(x)
        x = self.encoder(x)
        if isinstance(x, tuple):
            # batch_size*seq_size*hidden_size
            x, _ = x
        # batch_size*seq_size*label_num
        y = self.classify(x)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(y, target, mask, reduction="mean")
            else:
                # batch_size*seq_size*label_num => (batch_size*seq_size)*label_num
                return self.loss_func(y.view(-1, y.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(y)
            else:
                return y


def choose_optim(config, model):
    optim_type = config["optim_type"]
    learn_rate = config["learn_rate"]
    if optim_type.upper() == "SGD":
        return SGD(model.parameters(), lr=learn_rate)
    else:
        return Adam(model.parameters(), lr=learn_rate)


if __name__ == '__main__':
    from conf.setting import Config
    from core.data_generator import data_load

    Config["use_crf"] = True
    dg = data_load(Config, Config["train_data_path"], False)
    dg = dg.__iter__()
    batch_x, batch_y = dg.__next__()
    model = TorchModel(Config)
    loss = model(batch_x, batch_y)
    print(loss.item())
    # crf:219.058349609375
    # no_crf:2.1702370643615723
