# -*-coding:utf-8-*-
import os

# 项目文件夹地址
BASE_PATH = os.path.dirname(os.path.dirname(__file__))
# 模型训练配置文件
Config = {
    "train_data_path": r"D:\Text_Categorization_System_TCS\pre_train\data_resource\Train_seg.txt",
    "valid_data_path": r"D:\Text_Categorization_System_TCS\pre_train\data_resource\Valid_seg.txt",
    "test_data_path": r"D:\Text_Categorization_System_TCS\pre_train\data_resource\Test.txt",
    "dict_data_path": r"D:\Text_Categorization_System_TCS\pre_train\data_resource\dict.txt",
    "title_data_path": r"D:\Text_Categorization_System_TCS\pre_train\data_resource\title_set.json",
    "bert_pre_train_path": r"D:\Text_Categorization_System_TCS\pre_train\bert-base-chinese",
    "model_save_path": r"D:\Text_Categorization_System_TCS\pre_train\train_model_save",
    "picture_save_path": r"D:\Text_Categorization_System_TCS\pre_train\loss_picture",
    "model_type": "fast_text",
    "max_length": 30,
    "batch_size": 128,
    "hidden_size": 128,
    "kernel_size": 3,
    "hidden_layer_num": 2,
    "classify_num": 14,
    "pooling_type": "max",  # max,avg.cls
    "optim_type": "adam",
    "learn_rate": 1e-3,
    "epoch_time": 3,
    "seed": 768,
}

########################字典配置#####################################
# 定义三种日志输出格式 开始
standard_format = '[%(asctime)s][%(threadName)s:%(thread)d][task_id:%(name)s][%(filename)s:%(lineno)d]' \
                  '[%(levelname)s][%(message)s]'  # 其中name为getlogger指定的名字
simple_format = '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d]%(message)s'
id_simple_format = '[%(levelname)s][%(asctime)s] %(message)s'
# 获取项目的根目录root_path
logfile_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # log文件的目录
# 设置日志文件的文件名
logfile_name = 'all1.log'  # log文件名
# 存放日志文件的目录的创建，没有就创建
# 如果不存在定义的日志目录就创建一个
logfile_dir = os.path.join(logfile_dir, 'log')
if not os.path.isdir(logfile_dir):
    os.mkdir(logfile_dir)
# 设置日志文件的路径
# log文件的全路径
logfile_path = os.path.join(logfile_dir, logfile_name)
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
            'formatter': 'simple'
        },
        # 打印到文件的日志,收集info及以上的日志
        'default': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',  # 保存到文件
            'formatter': 'standard',
            'filename': logfile_path,  # 日志文件路径
            'maxBytes': 1024 * 1024 * 5,  # 日志大小 5M
            'backupCount': 5,  # 日志轮转文件数量
            'encoding': 'utf-8',  # 日志文件的编码，再也不用担心中文log乱码了
        },
    },
    'loggers': {  # 5. 设置日志对象
        # logging.getLogger(__name__)拿到的logger配置
        "test": {
            'handlers': ['console'],  # 这里把上面定义的两个handler都加上，即log数据既写入文件又打印到屏幕
            'level': 'DEBUG',
            'propagate': False,  # 向上（更高level的logger）传递
        },
        '': {
            'handlers': ['default', 'console'],  # 这里把上面定义的两个handler都加上，即log数据既写入文件又打印到屏幕
            'level': 'DEBUG',
            'propagate': False,  # 向上（更高level的logger）传递
        },
        # 可以自己配置固定名字但是你就要舍弃上面这种方式的配置，二者不共存
    },
}
