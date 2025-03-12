#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：TextGenerator_CausalLanguageModel 
@File    ：setting.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/3/4 15:41 
"""
import os

BASE_PATH = os.path.dirname(__file__)

Config = {
    "train_data_path": r"D:\TextGenerator_CausalLanguageModel\data_resource\corpus.txt",
    "vocab_path": r"D:\TextGenerator_CausalLanguageModel\data_resource\vocab.txt",
    "bert_model_path": r"D:\TextGenerator_CausalLanguageModel\bert-base-chinese",
    "train_data_size": 30000,
    "model_type": "bert",
    "batch_size": 30,
    "max_length": 10,
    "learning_rate": 1e-5,
    "optim_type": "Adam",
    "hidden_size": 256,
    "num_layers": 2,
    "seed": 768,
    "epoch": 20,
    "output_path": r"D:\TextGenerator_CausalLanguageModel\Output",
}

# 定义三种日志输出格式 开始
standard_format = '[%(asctime)s][%(threadName)s:%(thread)d][task_id:%(name)s][%(filename)s:%(lineno)d]' \
                  '[%(levelname)s][%(message)s]'  # 其中name为getlogger指定的名字
simple_format = '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d]%(message)s'
id_simple_format = '%(message)s'
# 获取项目的根目录root_path
log_path = os.path.join(BASE_PATH, "log")  # log文件的目录
# 设置日志文件的文件名
log_name = 'all.log'  # log文件名
# 存放日志文件的目录的创建，没有就创建
if not os.path.isdir(log_path):
    os.mkdir(log_path)
# 设置日志文件的路径
# log文件的全路径
log_file_path = os.path.join(log_path, log_name)
# log配置字典
LOGGING_DIC = {
    'version': 1,  # 1. 日志的版本号
    'disable_existing_loggers': False,  # 2. 关闭已存在的日志
    'formatters': {  # 3. 日志格式
        'standard': {
            'format': standard_format
        },
        'simple': {
            'format': simple_format
        },
        'test': {
            'format': id_simple_format
        }
    },
    'filters': {},
    'handlers': {  # 4. 日志的输出配置
        # 打印到终端的日志
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',  # 打印到屏幕
            'formatter': 'test'
        },
        # 打印到文件的日志,收集info及以上的日志
        'default': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',  # 保存到文件
            'formatter': 'test',
            'filename': log_file_path,  # 日志文件路径
            'maxBytes': 1024 * 1024 * 5,  # 日志大小 5M
            'backupCount': 5,  # 日志轮转文件数量
            'encoding': 'utf-8',  # 日志文件的编码，再也不用担心中文log乱码了
        },
    },
    'loggers': {  # 5. 设置日志对象
        'test': {
            'handlers': ['console', ],  # 这里把上面定义的两个handler都加上，即log数据既写入文件又打印到屏幕
            'level': 'DEBUG',
            'propagate': False,  # 向上（更高level的logger）传递
        },
        '': {
            'handlers': ['default', ],  # 这里把上面定义的两个handler都加上，即log数据既写入文件又打印到屏幕
            'level': 'DEBUG',
            'propagate': False,  # 向上（更高level的logger）传递
        },
    },
}
if __name__ == '__main__':
    from transformers import BertModel

