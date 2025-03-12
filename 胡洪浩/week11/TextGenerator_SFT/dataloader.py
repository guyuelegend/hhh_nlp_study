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
import json


class MyDataSet(Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained(self.config["bert_model_path"])
        self.vocab = self.load_vocab()
        if config["model_type"] == "bert":
            self.vocab = self.tokenizer.vocab
        self.config["vocab_size"] = len(self.vocab)

        self.load()

    def load(self):
        windows = self.config["max_length"]
        with open(self.data_path, mode='rt', encoding='utf-8') as f:
            for index, line in enumerate(f):
                data_dict = json.loads(line.strip())
                title, content = data_dict["title"], data_dict["content"]
                title = self.encode(title)
                content = self.encode(content)[1:]
                input_seq = title + content[:windows - len(title)]
                label_seq = content[:windows - len(title)]
                mask_seq = self.create_mask(label_seq, input_seq)
                label_seq = self.label_padding(label_seq, input_seq)

                input_seq = torch.LongTensor(input_seq)
                label_seq = torch.LongTensor(label_seq)
                # if index < 3:
                # print(input_seq, mask_seq, label_seq, sep="\n")
                # print(input_seq.shape, mask_seq.shape, label_seq.shape, sep="")
                self.data.append([input_seq, mask_seq, label_seq])
        return

    def encode(self, sentence):
        if self.config["model_type"].lower() == "bert":
            return self.tokenizer.encode(sentence,
                                         # pad_to_max_length=True,
                                         # padding="max_length",
                                         # max_length=self.config["max_length"],
                                         add_special_tokens=True)
        else:
            return [self.vocab.get(word, self.vocab.get("<UNK>")) for word in sentence]

    @staticmethod
    def create_mask(label_seq, input_seq):
        tril_length = len(label_seq)  # 3
        total_length = len(input_seq)  # 7
        a_mask = torch.ones(total_length - tril_length, total_length - tril_length)
        b_mask = torch.zeros(total_length - tril_length, tril_length)
        c_mask = torch.ones(tril_length, total_length - tril_length)
        d_mask = torch.tril(torch.ones(tril_length, tril_length))
        # print(a_mask, b_mask, sep='\n')
        mask_up = torch.cat((a_mask, b_mask), dim=-1)
        mask_down = torch.cat((c_mask, d_mask), dim=-1)
        return torch.cat((mask_up, mask_down), dim=0)

    @staticmethod
    def label_padding(label_seq, input_seq, ignore_idx=-100):
        input_len = len(input_seq)
        label_seq = label_seq + [ignore_idx]
        label_seq = [ignore_idx] * (input_len - len(label_seq)) + label_seq
        return label_seq

    def load_vocab(self):
        vocab = dict()
        with open(self.config["vocab_path"], mode='rt', encoding='utf-8') as f:
            for index, line in enumerate(f):
                vocab[line[:-1]] = index
        vocab["\n"] = len(vocab)
        vocab[" "] = len(vocab)
        return vocab

    def __len__(self):
        return len(self.data)

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
