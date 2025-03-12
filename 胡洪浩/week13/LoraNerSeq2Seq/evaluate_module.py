#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：LoraNerSeq2Seq 
@File    ：evaluate_module.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/3/11 14:06 
"""
import re
import numpy as np
import torch
from data_prepare import data_load
from collections import defaultdict


class Evaluator:
    def __init__(self, config, model, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.dg = data_load(config, config["valid_data_path"])
        self.tag = ("识别正确", "样本实体数", "识别实体数")
        self.state_dict = {"LOCATION": {}.fromkeys(self.tag, 0),
                           "ORGANIZATION": {}.fromkeys(self.tag, 0),
                           "PERSON": {}.fromkeys(self.tag, 0),
                           "TIME": {}.fromkeys(self.tag, 0), }

    def eval(self):
        print("测评开始".center(100, "="))
        self.model.eval()
        self.state_dict = {"LOCATION": {}.fromkeys(self.tag, 0),
                           "ORGANIZATION": {}.fromkeys(self.tag, 0),
                           "PERSON": {}.fromkeys(self.tag, 0),
                           "TIME": {}.fromkeys(self.tag, 0), }
        for index, batch_data in enumerate(self.dg):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, label_id = batch_data
            with torch.no_grad():
                # y = self.model(input_id)
                y = self.model(input_id)[0]
                y = torch.argmax(torch.softmax(y, dim=-1), dim=-1).squeeze()
            self.write_state(y, label_id)
        acc = self.show_state()
        print("测评结束".center(100, "="))
        return acc

    def write_state(self, y_pre, labels):
        assert len(y_pre) == len(labels)
        y_pre = y_pre.cpu().detach().tolist()
        labels = labels.cpu().detach().tolist()
        for y, label in zip(y_pre, labels):
            self.cal_state_dic(y, label)
        return

    def show_state(self):
        f1_scores = list()
        for key in self.state_dict.keys():
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            # self.tag = ("识别正确", "样本实体数", "识别实体数")
            precision = self.state_dict[key]["识别正确"] / (self.state_dict[key]["识别实体数"] + 1e-5)
            recall = self.state_dict[key]["识别正确"] / (self.state_dict[key]["样本实体数"] + 1e-5)
            f1 = 2 * precision * recall / (precision + recall + 1e-5)
            f1_scores.append(f1)
            print("{}类实体，准确率：{:2%}，召回率：{:2%}".format(key, precision, recall))
        print("Macro_F1: {:2%}".format(np.mean(f1_scores)))
        correct_pre = np.sum([self.state_dict[key]["识别正确"] for key in self.state_dict.keys()])
        total_pre = np.sum([self.state_dict[key]["识别实体数"] for key in self.state_dict.keys()])
        true_enti = np.sum([self.state_dict[key]["样本实体数"] for key in self.state_dict.keys()])
        micro_precision = correct_pre / (total_pre + 1e-5)
        micro_recall = correct_pre / (true_enti + 1e-5)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-5)
        print("Mirco_F1: {:2%}".format(micro_f1))
        return micro_f1

    def cal_state_dic(self, sequence, label):
        lab_res = defaultdict(list)
        seq_res = defaultdict(list)
        sentence = "".join(str(i) for i in range(len(lab_res)))
        seq = "".join(str(i) for i in sequence)
        lab = "".join(str(i) for i in label)
        for i in re.finditer(r"04+", seq):
            s, e = i.span()
            seq_res["LOCATION"].append(sentence[s:e])
        for i in re.finditer(r"04+", lab):
            s, e = i.span()
            lab_res["LOCATION"].append(sentence[s:e])
        for i in re.finditer(r"15+", seq):
            s, e = i.span()
            seq_res["ORGANIZATION"].append(sentence[s:e])
        for i in re.finditer(r"15+", lab):
            s, e = i.span()
            lab_res["ORGANIZATION"].append(sentence[s:e])
        for i in re.finditer(r"26+", seq):
            s, e = i.span()
            seq_res["PERSON"].append(sentence[s:e])
        for i in re.finditer(r"26+", lab):
            s, e = i.span()
            lab_res["PERSON"].append(sentence[s:e])
        for i in re.finditer(r"37+", seq):
            s, e = i.span()
            seq_res["TIME"].append(sentence[s:e])
        for i in re.finditer(r"37+", lab):
            s, e = i.span()
            lab_res["TIME"].append(sentence[s:e])
        for enti in self.state_dict.keys():
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            # self.tag = ("识别正确", "样本实体数", "识别实体数")
            for j in seq_res[enti]:
                self.state_dict[enti]["识别实体数"] += 1
                if j in lab_res[enti]:
                    self.state_dict[enti]["识别正确"] += 1
            for j in lab_res[enti]:
                self.state_dict[enti]["样本实体数"] += 1
        return


if __name__ == '__main__':
    lst = [4, 7, 4, 3, 8, 4, 2, 8, 4, 2, 2, 7, 6, 8, 6, 8, 8, 7, 4, 8, 2, 0, 7, 4,
           4, 4, 1, 2, 6, 8, 2, 4, 8, 4, 8, 4, 8, 4, 8, 8]
    s = "".join(str(i) for i in lst)
    for i in re.finditer("48+", s):
        print(i)
        print(i.span())
