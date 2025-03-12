#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：LoraNerSeq2Seq 
@File    ：train_module.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/3/11 12:00 
"""
import os
import logging
import os.path
from logging import config as cf
import random
import torch
import torch.nn as nn
import numpy as np
from setting import Config, LOGGING_DIC
from data_prepare import data_load
from model_module import choose_optim, NerModel, get_lora_model
from evaluate_module import Evaluator

cf.dictConfig(LOGGING_DIC)
testlogger = logging.getLogger("test")

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def run(config, save_flag=True):
    if not os.path.exists(config["model_save_path"]):
        os.mkdir(config["model_save_path"])
    # 数据准备
    dg = data_load(config, config["train_data_path"])
    # 模型准备
    # model = NerModel(config)
    model = get_lora_model(Config)
    if torch.cuda.is_available():
        model = model.cuda()
    # 优化器准备
    optimizer = choose_optim(config, model)
    # 模型测评
    evaluator = Evaluator(config, model, testlogger)
    # loss观察
    watch = []
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        watch_loss = []
        print("第{}次训练开始".format(epoch).center(100, "="))
        for index, batch_data in enumerate(dg):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, label_id = batch_data
            optimizer.zero_grad()
            loss = model(input_id)[0]
            loss = nn.CrossEntropyLoss()(loss.view(-1, loss.shape[-1]), label_id.view(-1))
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        avg_loss = np.sum(watch_loss) / len(watch_loss)
        acc = evaluator.eval()
        watch.append([acc, avg_loss])
    if save_flag:
        save_path = os.path.join(config["model_save_path"], "{}_lora.pth".format(config["model_type"]))
        torch.save(model.state_dict(), save_path)
    return


if __name__ == '__main__':
    run(Config)
