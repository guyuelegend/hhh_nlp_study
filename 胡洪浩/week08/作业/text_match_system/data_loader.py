# -*-coding:utf-8-*-
import torch
import jieba
import random
import json
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict


class MyData(Dataset):
    def __init__(self, config, data_path):
        self.label_dict_path = config["label_dict_path"]
        self.vocab_path = config["vocab_path"]
        self.max_length = config["max_length"]
        self.epoch_sample_size = config["epoch_sample_size"]
        self.positive_rate = config["positive_rate"]
        self.data_path = data_path
        self.knwb = defaultdict(list)
        self.data = []
        self.vocab = self.load_vocab()
        config["vocab_size"] = len(self.vocab)
        self.data_type = None
        self.label_dict = self.load_label_dic()
        self.load_data()

    def load_data(self):
        with open(self.data_path, mode='rt', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = "train"
                    sentences = line["questions"] + [line["target"]]
                    label_seq = self.label_dict.get(line["target"], 0)
                    for sentence in sentences:
                        sentence_seq = self.sentence_encode(sentence)
                        sentence_seq = torch.LongTensor(sentence_seq)
                        self.knwb[label_seq].append(sentence_seq)
                else:
                    self.data_type = "eval"
                    sentence_seq = self.sentence_encode(line[0])
                    label_seq = self.label_dict.get(line[1], 0)
                    x_seq = torch.LongTensor(sentence_seq)
                    label_seq = torch.LongTensor([label_seq])
                    self.data.append([x_seq, label_seq])

    def sentence_encode(self, sentence):
        if self.vocab_path == r"D:\TextMatchSystem\pre_data\data_resource\words.txt":
            res = [self.vocab.get(word, self.vocab["[UNK]"]) for word in jieba.cut(sentence)]
        else:
            res = [self.vocab.get(word, self.vocab["[UNK]"]) for word in sentence]
        res = res[:self.max_length]
        if len(res) < self.max_length:
            res += [0] * (self.max_length - len(res))
        return res

    def load_vocab(self):
        vocab = defaultdict(int)
        with open(self.vocab_path, mode='rt', encoding='utf-8') as f:
            for index, line in enumerate(f):
                word = line.strip()
                vocab[word] += index + 1
        return vocab

    def load_label_dic(self):
        with open(self.label_dict_path, mode='rt', encoding='utf-8') as f:
            label_dict = json.load(f)
        return label_dict

    def __getitem__(self, item):
        if self.data_type == "train":
            return self.sample_pair_generator()
        else:
            assert self.data_type == "eval", self.data_type
            return self.data[item]

    def __len__(self):
        if self.data_type == "train":
            return self.epoch_sample_size
        else:
            assert self.data_type == "eval", self.data_type
            return len(self.data)

    # def sample_pair_generator(self):
    #     standard_question_index = list(self.knwb.keys())
    #     if random.random() < self.positive_rate:
    #         p = random.choice(standard_question_index)
    #         if len(self.knwb[p]) >= 2:
    #             s1, s2 = random.sample(self.knwb[p], 2)
    #         else:
    #             return self.sample_pair_generator()
    #         return [s1, s2, torch.LongTensor([1])]
    #     else:
    #         p, n = random.sample(standard_question_index, 2)
    #         s1 = random.choice(self.knwb[p])
    #         s2 = random.choice(self.knwb[n])
    #         return [s1, s2, torch.LongTensor([-1])]
    def sample_pair_generator(self):  # triplet
        standard_question_index = list(self.knwb.keys())
        p, n = random.sample(standard_question_index, 2)
        if len(self.knwb[p]) >= 2:
            s1, s2 = random.sample(self.knwb[p], 2)
        else:
            return self.sample_pair_generator()
        if len(self.knwb[n]) >= 1:
            s3 = random.choice(self.knwb[n])
            return [s1, s2, s3]
        else:
            return self.sample_pair_generator()


def data_load(config, data_path, shuffle_flag=True):
    data_set = MyData(config, data_path)
    dg = DataLoader(data_set, shuffle=shuffle_flag, batch_size=config["batch_size"])
    return dg


if __name__ == '__main__':
    from conf.setting import Config

    Config["vocab_path"] = r"D:\TextMatchSystem\pre_data\data_resource\words.txt"
    dg = data_load(Config, Config["train_data_path"])
    # dg = data_load(Config, Config["valid_data_path"])
    # for i in dg.dataset.knwb.items():
    #     print(i)
    #     print('\n\n')
    sample = dg.__iter__().__next__()
    print(len(sample[0]), len(sample[0][0]))
    print(len(sample[1]), len(sample[1][0]))
    print(len(sample[2]), len(sample[2][0]))
