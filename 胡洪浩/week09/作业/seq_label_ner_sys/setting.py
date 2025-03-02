#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：SeqLabelNerSystem 
@File    ：setting.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/2/25 13:46 
"""
import os

Config = {
    "vocab_path": r"D:\SeqLabelNerSystem\db\model_resource\chars.txt",
    "train_data_path": r"D:\SeqLabelNerSystem\db\model_resource\train",
    "valid_data_path": r"D:\SeqLabelNerSystem\db\model_resource\test",
    "label_dict_path": r"D:\SeqLabelNerSystem\db\model_resource\schema.json",
    "model_save_path": r"D:\SeqLabelNerSystem\db\output",
    "bert_model_path": r"D:\SeqLabelNerSystem\db\bert-base-chinese",
    "model_type": "bert",
    "batch_size": 16,
    "max_length": 100,
    "hidden_size": 256,
    "label_num": 9,
    "use_crf": False,
    "num_layer": 2,
    "optim_type": "adam",
    "learn_rate": 1e-3,
    "epoch_time": 20,
    "seed": 768,
}
