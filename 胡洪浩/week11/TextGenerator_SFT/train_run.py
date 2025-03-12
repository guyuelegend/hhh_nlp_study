#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：TextGenerator_CausalLanguageModel
@File    ：models.py
@IDE     ：PyCharm
@Author  ：Guyuelegend
@Date    ：2025/3/4 17:28
"""
import os.path
import random
import logging
from logging import config as cf
import torch
import numpy as np
from setting import Config, LOGGING_DIC
from dataloader import generate_data
from models import CausalLanguageModel, choose_optim
from evaluate_run import Evaluate

cf.dictConfig(LOGGING_DIC)
test_logger = logging.getLogger("test")
main_logger = logging.getLogger("main")

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)


def run(config, save_flag=True):
    # flag表示cuda
    flag = torch.cuda.is_available()
    # 数据的准备
    dg = generate_data(config, config["train_data_path"])
    # 模型定义
    model = CausalLanguageModel(config).cuda() if flag else CausalLanguageModel(config)
    # 优化器定义
    optimizer = choose_optim(config, model)
    # 训练后测评定义
    evaluator = Evaluate(Config, model, dg.dataset.vocab, test_logger)
    # 记录loss
    watch = []
    # 定义训练
    for epoch in range(config["epoch"]):
        epoch += 1
        test_logger.info("第{}次训练开始".format(epoch).center(100, "="))
        loss_watch = []
        model.train()
        for index, batch_data in enumerate(dg):
            if flag:
                batch_data = [d.cuda() for d in batch_data]
            data_x, data_mask, data_y = batch_data
            optimizer.zero_grad()
            loss = model(data_x, data_y, data_mask)
            loss.backward()
            optimizer.step()
            loss_watch.append(loss.item())
        avg_loss = np.sum(loss_watch) / len(loss_watch)
        test_logger.info("第{}次平均loss: {}".format(epoch, avg_loss))
        pre_sentence = evaluator.eval("北京明年拟推工作日半价观看电影")
        watch.append([avg_loss, pre_sentence])
        main_logger.info("第{}次训练：".format(epoch) + pre_sentence + "loss: {}".format(avg_loss))
    # loss迭代展示
    print(watch)
    # 模型保存
    if save_flag:
        save_path = os.path.join(config["output_path"], "{}.pth".format(config["model_type"]))
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    test_logger.info("测试一下！")
    run(Config)
