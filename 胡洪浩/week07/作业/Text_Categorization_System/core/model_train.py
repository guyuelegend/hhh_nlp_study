# -*-coding:utf-8-*-
import logging
import random
import numpy as np
import torch
import numpy
import os
import pandas as pd
import openpyxl
from conf.setting import Config
from core.data_loader import load_data
from db.models import choose_optim, TorchModel
from core.evaluate_loader import Evaluator
from core.DataVisualization import loss_visual
from logging import config
from conf.setting import LOGGING_DIC

config.dictConfig(LOGGING_DIC)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger('test')

# 随机种子固定，有助于模型训练的复现
seed = Config["seed"]
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train_main(config):
    # 创建模型保存的路径
    if not os.path.isdir(config["model_save_path"]):
        os.mkdir(config["model_save_path"])
    # 数据的准备
    dg = load_data(config, config["train_data_path"])
    # 模型的建立
    model = TorchModel(config)
    # 优化器的选择
    optimizer = choose_optim(config, model)
    # 模型测试器选择
    evaluator = Evaluator(config, model)
    # cuda权限状态
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    # 训练结果的收集
    watch = []
    # 记录日志，输出训练超参数配置
    logger.info(
        "\n" + \
        "train parameter config".center(100, '-') + \
        "\n参数信息\n" + \
        "".center(100, '-')
    )
    print(" {} ".format(config["model_type"]).center(100, '='))
    # 定义迭代流程
    for epoch in range(config["epoch_time"]):
        epoch += 1
        watch_loss = []
        model.train()
        print("第{}次训练开始了......".format(epoch))
        for index, batch_data in enumerate(dg):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            x_data, y_data = batch_data
            optimizer.zero_grad()
            loss = model(x_data, y_data)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        loss_avg = np.mean(watch_loss)
        print("第{}次训练的平均loss:{}".format(epoch, loss_avg))
        acc = evaluator.load(epoch)
        watch.append((acc, loss_avg))
    print("".center(100, "="))
    # 训练残差与正确率可视化
    picture_save_path = r"{}\{}_train_loss.jpg".format(Config["picture_save_path"], Config["model_type"])
    loss_visual(watch, picture_save_path)
    # 保存模型
    save_path = os.path.join(config["model_save_path"], "{}_train.pth".format(config["model_type"]))
    torch.save(model.state_dict(), save_path)
    return acc


def test_model(model):
    pass


if __name__ == '__main__':
    acc = train_main(Config)
    excel_flag = bool(openpyxl.__version__)
    # test_dic = {
    #     "model_type": ["cnn", "rnn", "bert"],
    #     "learn_rate": [1e-3, 1e-4, 1e-5],
    #     "hidden_size": [128, 256, 768],
    #     "pooling_type": ["max", "avg"],
    #     "batch_size": [64, 128],
    # }
    # data = {key: [] for key in test_dic.keys()}
    # data["acc"] = []
    # print(data)
    # for mode_type in test_dic["model_type"]:
    #     Config["mode_type"] = mode_type
    #     for learn_rate in test_dic["learn_rate"]:
    #         Config["learn_rate"] = learn_rate
    #         for hidden_size in test_dic["hidden_size"]:
    #             Config["hidden_size"] = hidden_size
    #             for pooling_type in test_dic["pooling_type"]:
    #                 Config["pooling_type"] = pooling_type
    #                 for batch_size in test_dic["batch_size"]:
    #                     Config["batch_size"] = batch_size
    #                     data['model_type'].append(mode_type)
    #                     data['learn_rate'].append(learn_rate)
    #                     data['hidden_size'].append(hidden_size)
    #                     data['pooling_type'].append(pooling_type)
    #                     data['batch_size'].append(batch_size)
    #                     data['acc'].append(0.015649)
    # print(data)
    # excel_save_path = r"D:\NLP2025\test.xlsx"
    # pd.DataFrame(data).to_excel(excel_save_path, index=False, sheet_name='Test')
