#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：SeqLabelNerSystem 
@File    ：train_process.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/2/25 13:50 
"""
import random
import os
import numpy.random
import torch
import numpy as np
import pandas as pd
from conf.setting import Config
from core.data_generator import data_load
from db.models import TorchModel, choose_optim
from core.evaluate_process import Evaluate

# 固定随机种子，方便复现模型训练
seed = Config["seed"]
random.seed(seed)
torch.manual_seed(seed)
numpy.random.seed(seed)
torch.cuda.manual_seed(seed)


def run(config):
    # 数据加载
    dg = data_load(config, config["train_data_path"])
    # 模型的选择
    model = TorchModel(config)
    # cuda权限
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    # 优化器的选择
    optimizer = choose_optim(config, model)
    # 模型测评
    evaluator = Evaluate(config,model)
    # 残差记录
    watch = []
    # 开始训练
    for epoch in range(config["epoch_time"]):
        epoch += 1
        print("第{}次训练开始......".format(epoch))
        model.train()
        watch_loss = []
        for index, batch_data in enumerate(dg):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_seq, label_seq = batch_data
            optimizer.zero_grad()
            loss = model(input_seq, label_seq)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
            if index < 2:
                print(loss.item())
        loss_avg = np.sum(watch_loss) / len(watch_loss)
        print("第{}次的loss: {}".format( epoch, loss_avg))
        acc = evaluator.eval()
        watch.append([acc, loss_avg])
    # 可视化

    # 模型的保存
    model_save_path = config["model_save_path"]
    model_save_path = os.path.join(model_save_path, "{}.pth".format("ner"))
    torch.save(model.state_dict(), model_save_path)
    return acc


if __name__ == '__main__':
    acc = run(Config)
    print(acc)
