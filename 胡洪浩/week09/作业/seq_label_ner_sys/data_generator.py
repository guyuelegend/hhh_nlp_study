#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：SeqLabelNerSystem 
@File    ：data_generator.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/2/25 13:50 
"""
import json
import torch
import jieba
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class MyDataSet(Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.data = []
        self.sentences = []
        self.vocab = self.load_vocab()
        self.config["vocab_size"] = len(self.vocab)
        self.label_dict = self.load_label_dict()
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.load()

    def load(self):
        data_path = self.data_path
        sentence = []
        seq_label = []
        with open(data_path, mode='rt', encoding='utf-8') as f:
            for index,line in enumerate(f):
                line = line.strip()
                if len(line) == 0:
                    label_id = [self.label_dict.get(label, 8) for label in seq_label]
                    label_id = self.padding(label_id, pad_token=8)
                    if self.config["model_type"] == "bert":
                        input_id = self.tokenizer.encode(''.join(sentence), max_length=self.config["max_length"],
                                                         pad_to_max_length=True)
                        label_id.insert(0, 8)
                        label_id.pop(-1)
                        # if index < 200:
                        #     print(input_id,''.join(sentence),sep='\n')
                    else:

                        input_id = self.encoder_sentence(''.join(sentence))

                    label_id = torch.LongTensor(label_id)
                    input_id = torch.LongTensor(input_id)
                    self.data.append([input_id, label_id])
                    self.sentences.append(''.join(sentence))
                    sentence.clear()
                    seq_label.clear()
                    continue
                word, label = line.split(' ')
                sentence.append(word)
                seq_label.append(label)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def load_vocab(self):
        path = self.config["vocab_path"]
        vocab = {}
        with open(path, mode='rt', encoding='utf-8') as f:
            for index, line in enumerate(f):
                word = line.strip()
                vocab[word] = index + 1
        return vocab

    def load_label_dict(self):
        path = self.config["label_dict_path"]
        with open(path, mode='rt', encoding='utf-8') as f:
            label_dict = json.load(f)
        return label_dict

    def encoder_sentence(self, sentence, padding: bool = True):
        seq = [self.vocab.get(word, self.vocab["[UNK]"]) for word in sentence]
        if padding:
            seq = self.padding(seq)
        return seq

    def padding(self, inputs, pad_token=0):
        inputs = inputs[:self.config["max_length"]]
        if len(inputs) < self.config["max_length"]:
            inputs += [pad_token] * (self.config["max_length"] - len(inputs))
        return inputs


def data_load(config, data_path, flag=True):
    ds = MyDataSet(config, data_path)
    dg = DataLoader(ds, batch_size=config["batch_size"], shuffle=flag)
    return dg


if __name__ == '__main__':
    from conf.setting import Config

    dg2 = data_load(Config, Config["train_data_path"],False)
    dg2 = dg2.__iter__()
    # batch_size * seq_size
    inp, lab = next(dg2)
    print(inp.shape)
    print(inp)
    print(lab.shape)
    for i in lab:
        print(i)
