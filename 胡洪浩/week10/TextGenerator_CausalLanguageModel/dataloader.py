#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：TextGenerator_CausalLanguageModel 
@File    ：dataloader.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/3/4 15:44 
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import time


class MyDataSet(Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained(self.config["bert_model_path"])
        self.vocab = self.load_vocab()
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        windows = self.config["max_length"]
        count = 0
        train_data = self.load_corpus(self.data_path)
        if len(train_data) < windows:
            return
        while count + windows + 1 < len(train_data):
            input_seq = train_data[count:count + windows]
            label_seq = train_data[count + 1:count + windows + 1]
            input_seq = self.encode(input_seq)
            label_seq = self.encode(label_seq)
            # if count < 2:
            #     print(input_seq)
            #     print(label_seq)
            input_seq = torch.LongTensor(input_seq)
            label_seq = torch.LongTensor(label_seq)
            self.data.append([input_seq, label_seq])
            count += 1
        return

    def encode(self, sentence):
        if self.config["model_type"].lower() == "bert":
            return self.tokenizer.encode(sentence,
                                         # pad_to_max_length=True,
                                         padding="max_length",
                                         max_length=self.config["max_length"],
                                         add_special_tokens=False)
        else:
            return [self.vocab.get(word, self.vocab.get("<UNK>")) for word in sentence]

    @staticmethod
    def load_corpus(corpus_path):
        corpus = ""
        with open(corpus_path, mode='rt', encoding='gbk') as f:
            for index, line in enumerate(f):
                if index < 20000:
                    corpus += line
        return corpus

    def load_vocab(self):
        vocab = {"padding": 0}
        with open(self.config["vocab_path"], mode='rt', encoding='utf-8') as f:
            for index, line in enumerate(f):
                vocab[line[:-1]] = index + 1
        vocab["\n"] = len(vocab)
        if self.config["model_type"] == "bert":
            return self.tokenizer.vocab
        return vocab

    def __len__(self):
        return self.config["train_data_size"]

    def __getitem__(self, item):
        return self.data[item]


def generate_data(config, data_path, flag=True):
    ds = MyDataSet(config, data_path)
    dg = DataLoader(ds, batch_size=config["batch_size"], shuffle=flag)
    return dg


if __name__ == '__main__':
    from setting import Config

    start = time.time()
    data_g = generate_data(Config, Config["train_data_path"], False)
    data_g = data_g.__iter__()
    print("耗时：", time.time() - start, "s", sep='')
    print(next(data_g))
    print(next(data_g))
""""""
