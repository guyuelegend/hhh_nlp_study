#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：SeqLabelNerSystem 
@File    ：evaluate_process.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/2/25 13:50 
"""
import torch
import numpy as np
from core.data_generator import data_load
from collections import defaultdict
import re


class Evaluate:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.valid_data = data_load(config,
                                    config["valid_data_path"], False)
        self.stats_dict = {}

    def eval(self):
        """
        1. 定义一个有机构，时间，人名，机构的统计字典
        2. 模型数据，拿到测试的概率分布
        """
        print("".center(100, "="))
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}

        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_seq, label_seq = batch_data
            with torch.no_grad():
                pred_results = self.model(input_seq)  # 不输入labels，使用模型当前参数进行预测
            self.write_stats(label_seq, pred_results, sentences)
        acc = self.show_stats()
        print("".center(100, "="))
        return acc

    def write_stats(self, labels, pred_results, sentences):
        """

        """
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):

            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)# 真实有多少个实体数
            pred_entities = self.decode(sentence, pred_label)# 预测有多少个实体数
            # print("=+++++++++")
            # print(true_entities)
            # print(pred_entities)
            # print('=+++++++++')
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        """
        微观F1根据命名实体分开算，
        宏观F1不管命名实体分类统一算
        :return:
        """
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            print("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        print("Micro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        print("Macro-F1 %f" % micro_f1)
        return micro_f1

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''

    def decode(self, sentence, labels):
        """

        """
        if self.config["model_type"] == "bert":
            sentence = "$" + sentence
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results
