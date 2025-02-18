# -*- coding:utf-8-*-
import torch
import json
from torch.utils.data import DataLoader, Dataset
from conf.setting import Config
from transformers import BertTokenizer


class MyDataSet(Dataset):
    def __init__(self, config, data_path):
        self.data_path = data_path
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_pre_train_path"])
        self.tag_dict = self.load_tag()
        self.vocab = self.load_vocab()
        self.config['vocab_size'] = len(self.vocab)
        self.data = []
        self.load()

    def load(self):
        with open(self.data_path, mode='rt', encoding='utf-8') as f:
            for line in f:
                tag = line[:5].strip()[-2:]
                title = line[5:].strip()
                if self.config["model_type"] == "bert":
                    input_seq = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                      pad_to_max_length=True)
                else:
                    input_seq = self.text2seq(title)
                label_seq = self.tag_dict.get(tag, 0)
                input_seq = torch.LongTensor(input_seq)
                label_seq = torch.LongTensor([label_seq])
                self.data.append([input_seq, label_seq])
        return

    def load_tag(self):
        with open(self.config["title_data_path"], mode='rt', encoding='utf-8') as f:
            tag_dict = json.load(f)
        return tag_dict

    def load_vocab(self):
        with open(self.config["dict_data_path"], mode='rt', encoding='utf-8') as f:
            vocab = dict()
            for index, line in enumerate(f):
                char = line.strip()
                if vocab.get(char) is None:
                    vocab[char] = index + 1
        return vocab

    def text2seq(self, text):
        seq = [self.vocab.get(char, self.vocab["<unk>"]) for char in text]
        seq = seq[:self.config["max_length"]]
        if len(seq) < self.config["max_length"]:
            seq += [0] * (self.config["max_length"] - len(seq))
        return seq

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def load_data(config, data_path, optim=True):
    data_obj = MyDataSet(config, data_path)
    return DataLoader(data_obj, batch_size=config["batch_size"], shuffle=optim)


if __name__ == '__main__':
    dg = load_data(Config, Config["valid_data_path"])
    print("data: ", dg.__iter__().__next__())
    print("batch_size: ", dg.__iter__().__next__()[0].__len__())
    print("max_length: ", dg.__iter__().__next__()[0][-1].__len__())
