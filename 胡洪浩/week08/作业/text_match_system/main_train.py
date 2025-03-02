# -*-coding:utf-8-*-
import random
import os
import torch
import numpy as np
import pandas as pd
import logging
from conf.setting import Config
from core.data_loader import data_load
from db.models import SiameseNetwork, choose_optimizer
from main_evaluator import Evaluate
from core.loss_visualize import loss_visual

# 固定随机因子，方便模型训练复现
seed = Config["seed"]
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)


def main_run(config):
    # cuda的使用权
    cuda_flag = torch.cuda.is_available()
    # 定义数据加载器
    data_generator = data_load(config, config["train_data_path"])
    # 模型的定义
    model = SiameseNetwork(config)
    if cuda_flag:
        model = model.cuda()
    # 优化器的定义
    optimizer = choose_optimizer(config, model)
    # 模型评价
    evaluator = Evaluate(config, model)
    # 训练残差图数据收集
    watch = []
    # 迭代
    for epoch in range(config["epoch_time"]):
        epoch += 1
        watch_loss = []
        print("第{}次开始训练了......".format(epoch))
        model.train()
        for index, batch_data in enumerate(data_generator):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id1, input_id2, label = batch_data
            optimizer.zero_grad()
            loss = model(input_id1, input_id2, label)
            loss.backward()
            optimizer.step()
            if index < 2:
                print(loss.item())
            watch_loss.append(loss.item())
        print("第{}次loss:{}".format(epoch, np.sum(watch_loss)))
        acc = evaluator.eval()
        watch.append([acc, np.sum(watch_loss) / len(watch_loss)])
    # 模型残差图输出
    p_save_path = r"D:\TextMatchSystem\pre_data\loss_picture"
    p_save_path = os.path.join(p_save_path, "{}".format("111"))
    loss_visual(watch, p_save_path)
    # 模型保存
    model_save_path = config["model_save_path"]
    model_save_path = os.path.join(model_save_path, "{}.pth".format(config["model_save_name"]))
    torch.save(model.state_dict(), model_save_path)
    return


if __name__ == '__main__':
    main_run(Config)
