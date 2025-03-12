#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：LoraNerSeq2Seq 
@File    ：data_prepare.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/3/11 9:22 
"""
import json
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset


class NerDataset(Dataset):
    def __init__(self, config, data_path):
        self.data_path = data_path
        self.config = config
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = self.load_vocab()
        config["vocab_size"] = len(self.vocab)
        self.label_dict = self.load_label_dict()
        config["label_size"] = len(self.label_dict)

        self.load()

    def load(self):
        sentence_seq = []
        label_seq = []
        with open(self.data_path, mode='rt', encoding='utf-8') as f:
            for index, line in enumerate(f):
                if line == '\n':
                    sentence_seq = self.padding(sentence_seq)
                    input_id = torch.LongTensor(sentence_seq)
                    label_seq = self.padding(label_seq, pad=self.label_dict["O"])
                    label_id = torch.LongTensor(label_seq)
                    self.data.append([input_id, label_id])
                    sentence_seq = []
                    label_seq = []
                    continue
                char, tag = line[:-1].split(" ")
                label_seq.append(self.label_dict.get(tag, self.label_dict["O"]))
                sentence_seq.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return

    def padding(self, seq, pad=0):
        windows_size = self.config["max_length"]
        seq = seq[:windows_size]
        if len(seq) < windows_size:
            seq += [pad] * (windows_size - len(seq))
        return seq

    def load_vocab(self):
        vocab_path = self.config["vocab_path"]
        vocab = dict()
        with open(vocab_path, mode='rt', encoding='utf-8') as f:
            for index, line in enumerate(f):
                vocab[line[:-1]] = index + 1
            vocab["[PAD]"] = 0
        if self.config["model_type"] == 'bert':
            vocab = self.tokenizer.vocab
        return vocab

    def load_label_dict(self):
        label_path = self.config["label_path"]
        with open(label_path, mode='rt', encoding='utf-8') as f:
            label_dict = json.load(f)
        return label_dict

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def data_load(config, data_path, flag=True):
    ds = NerDataset(config, data_path)
    dg = DataLoader(ds, config["batch_size"], shuffle=flag)
    return dg


if __name__ == '__main__':
    from setting import Config

    dgs = data_load(Config, Config["train_data_path"], False)
    for indexs, batch_data in enumerate(dgs):
        x, y = batch_data
        if indexs < 2:
            print(x)
            print(x.shape)
            print(y)
            print(y.shape)
