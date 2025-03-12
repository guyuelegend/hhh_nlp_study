#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：LoraNerSeq2Seq 
@File    ：model_module.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/3/11 10:22 
"""
import json
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification
from torchcrf import CRF
from torch.optim import SGD, Adam
from setting import Config
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig


class NerModel(nn.Module):
    def __init__(self, config):
        super(NerModel, self).__init__()
        self.config = config
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        label_size = config["label_size"]
        if config["model_type"] == 'bert':
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
            hidden_size = 2 * hidden_size
        self.classifier = nn.Linear(hidden_size, label_size)
        self.crf = CRF(label_size, batch_first=True)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        # batch_size * seq_size * hidden_size
        if self.config["model_type"] == "bert":
            mask = torch.ones(x.shape[0], x.shape[1], x.shape[1])
            x = self.encoder(x, attention_mask=mask)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        if isinstance(x, tuple):
            x, _ = x
        # batch_size * seq_size * vocab_size
        y_pre = self.classifier(x)
        if target is not None:
            if self.config["use_crf"]:
                mask = target.gt(-1).bool()
                # print(type(mask), type(mask.bool()))
                return - self.crf(y_pre, target, mask, reduction="mean")
            else:
                return self.loss_func(y_pre.view(-1, y_pre.shape[-1]), target.view(-1))
        else:
            if self.config["use_crf"]:
                return torch.LongTensor(self.crf.decode(y_pre))
            else:
                # 不是文本生成任务，不定要输出概率分布进行beam_size，直接输出标签结果更为方便
                y_pre = nn.functional.softmax(y_pre, dim=-1)
                y_pre = torch.argmax(y_pre, dim=-1).squeeze()
                return y_pre


def choose_optim(config, model):
    learning_rate = config["learning_rate"]
    optim_type = config["optim_type"].lower()
    if optim_type == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    else:
        return Adam(model.parameters(), lr=learning_rate)


def load_label_dict(config):
    label_path = config["label_path"]
    with open(label_path, mode='rt', encoding='utf-8') as f:
        label_dict = json.load(f)
    return label_dict


def get_lora_model(config):
    pretrain_ner_model = AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"], num_labels=9)
    pretrain_ner_model.config.label2id = load_label_dict(Config)
    pretrain_ner_model.config.id2label = {str(v): k for k, v in load_label_dict(Config).items()}
    torch.nn.init.normal_(pretrain_ner_model.classifier.bias, mean=0.0, std=0.02)
    torch.nn.init.normal_(pretrain_ner_model.classifier.weight, mean=0.0, std=0.02)
    # print(vars(PretrainNerModel))
    # print(PretrainNerModel.config)
    model = pretrain_ner_model
    # fine_tuning_strategy
    fine_tuning_tactics = config["tuning_tactics"].lower()
    peft_config = None
    if fine_tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=64,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense"]
        )
    elif fine_tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif fine_tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif fine_tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    model = get_peft_model(model, peft_config)
    # print(model.state_dict().keys())
    # for i in model.state_dict().keys():
    #     print(i)
    # 最后的线性层开放梯度计算
    for para in model.get_submodule("model").get_submodule("classifier").parameters():
        para.requires_grad = True
    return model


if __name__ == '__main__':
    from setting import Config
    from data_prepare import data_load

    # dg = data_load(Config, Config["train_data_path"], False)
    # dg = next(dg.__iter__())
    # x, y = dg
    # models = NerModel(Config)
    # loss = models(x, y)
    # print(loss.item())
    # # crf.decode输出的是batch_size * seq_size的列表
    # y = models(x)
    # for i in y:
    #     # if not Config["use_crf"]:
    #     #     i = torch.argmax(i, dim=-1).view(-1)
    #     print(i)

    get_lora_model(Config)
"""
  "id2label": {
    "0": "B-LOCATION",
    "1": "B-ORGANIZATION",
    "2": "B-PERSON",
    "3": "B-TIME",
    "4": "I-LOCATION",
    "5": "I-ORGANIZATION",
    "6": "I-PERSON",
    "7": "I-TIME",
    "8": "O"
  },
  "label2id": {
    "B-LOCATION": 0,
    "B-ORGANIZATION": 1,
    "B-PERSON": 2,
    "B-TIME": 3,
    "I-LOCATION": 4,
    "I-ORGANIZATION": 5,
    "I-PERSON": 6,
    "I-TIME": 7,
    "O": 8
  },
"""
