# -*-coding:utf-8-*-
import os

Config = {
    "label_dict_path": r"D:\TextMatchSystem\pre_data\data_resource\schema.json",
    "train_data_path": r"D:\TextMatchSystem\pre_data\data_resource\train.json",
    "valid_data_path": r"D:\TextMatchSystem\pre_data\data_resource\valid.json",
    "vocab_path": r"D:\TextMatchSystem\pre_data\data_resource\chars.txt",
    "model_save_path": r"D:\TextMatchSystem\pre_data\models_save",
    "model_save_name": "111",
    "epoch_sample_size": 3000,
    "positive_rate": 0.5,
    "max_length": 30,
    "batch_size": 64,
    "hidden_size": 128,
    "optimizer_type": "adam",
    "learn_rate": 1e-3,
    "epoch_time": 20,
    "seed": 768,
}
