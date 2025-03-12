#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：TextGenerator_CausalLanguageModel
@File    ：models.py
@IDE     ：PyCharm
@Author  ：Guyuelegend
@Date    ：2025/3/4 17:28
"""
import random
from transformers import BertTokenizer
import torch
import numpy as np


class Evaluate:
    def __init__(self, config, model, vocab, logger):
        self.config = config
        self.model = model
        self.vocab = vocab
        self.reverse_vocab = {str(v): k for k, v in vocab.items()}
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])

    def eval(self, cls_token):
        self.model.eval()
        self.logger.info("模型测评开始".center(100, "="))
        pre_str = ""
        ret = ""
        x_lst = self.encode(cls_token)
        while pre_str != "\n" and len(ret) <= 50:
            ret += pre_str

            x = torch.LongTensor(x_lst).unsqueeze(0)
            with torch.no_grad():
                if torch.cuda.is_available():
                    x = x.cuda()
                y = self.model(x)[0][-1]
            index = int(torch.argmax(y, dim=-1))
            x_lst += [index]
            pre_str = self.reverse_vocab.get(str(index), "N/A")
        self.logger.info(cls_token + "[SEP]" + ret)
        self.logger.info("end".center(100, "="))
        return ret

    @staticmethod
    def sampling_strategy(prob_distribution):
        if random.random() > 0.1:
            # strategy = "greedy"
            return int(torch.argmax(prob_distribution, dim=-1)[-1])
        else:
            # strategy = "sample"
            prob_distribution = prob_distribution.to_numpy()
            # np.random.choice中是定义input一维矩阵中选取每一个元素的概率，我们这样写有一点巧思在里面
            return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution).item()

    def encode(self, sentence):
        # sentence = sentence[-self.config["max_length"]:]
        if self.config["model_type"].lower() == "bert":
            return self.tokenizer.encode(sentence,
                                         # pad_to_max_length=True,
                                         # padding="max_length",
                                         # max_length=self.config["max_length"],
                                         add_special_tokens=True)
        else:
            return [self.vocab.get(word, self.vocab.get("<UNK>")) for word in sentence]


def load_vocab(config):
    vocab = {"padding": 0}
    with open(config["vocab_path"], mode='rt', encoding='utf-8') as f:
        for index, line in enumerate(f):
            vocab[line[:-1]] = index + 1
    vocab["\n"] = len(vocab)
    return vocab


if __name__ == '__main__':
    from setting import Config, LOGGING_DIC
    from dataloader import generate_data
    from models import CausalLanguageModel
    import logging
    from logging import config as cf

    cf.dictConfig(LOGGING_DIC)
    test_logger = logging.getLogger("test")
    PATH = r"D:\TextGenerator_CausalLanguageModel\Output\bert.pth"
    # dg = generate_data(Config, Config["train_data_path"])
    vocab = load_vocab(Config)
    Config["vocab_size"] = len(vocab)
    model = CausalLanguageModel(Config)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(PATH, map_location=device,weights_only=True))
    if torch.cuda.is_available():
        model = model.cuda()
    ev = Evaluate(Config, model, vocab, test_logger)
    ev.eval("让他在半年之前，就不能做出")
    ev.eval("李慕站在山路上，深深的呼吸")
    ev.eval("李慕它深呼一口气")
