#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：LoraNerSeq2Seq 
@File    ：predicate_module.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/3/11 17:51 
"""
import re
import json
import torch
from setting import Config
from model_module import get_lora_model
from transformers import BertTokenizer
from collections import defaultdict

model_path = r"D:\LoraNerSeq2Seq\Output\bert_lora.pth"
tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])
model = get_lora_model(Config)
# model = torch.load(model_path,weights_only=False)
model.load_state_dict(torch.load(model_path, weights_only=False))


def encode(sentence):
    return tokenizer.encode(sentence, add_special_tokens=False)


def load_label_dict(config):
    with open(config["label_path"], mode='rt', encoding='utf-8') as f:
        label_dict = json.load(f)
    return {str(v): k for k, v in label_dict.items()}


def show(sentence, label_seq):
    lab_res = defaultdict(list)
    lab = "".join(str(i) for i in label_seq)
    for i in re.finditer(r"04+", lab):
        s, e = i.span()
        if sentence[s:e] not in lab_res["LOCATION"]:
            lab_res["LOCATION"].append(sentence[s:e])
    for i in re.finditer(r"15+", lab):
        s, e = i.span()
        if sentence[s:e] not in lab_res["ORGANIZATION"]:
            lab_res["ORGANIZATION"].append(sentence[s:e])
    for i in re.finditer(r"26+", lab):
        s, e = i.span()
        if sentence[s:e] not in lab_res["PERSON"]:
            lab_res["PERSON"].append(sentence[s:e])
    for i in re.finditer(r"37+", lab):
        s, e = i.span()
        if sentence[s:e] not in lab_res["TIME"]:
            lab_res["TIME"].append(sentence[s:e])
    return lab_res


if __name__ == '__main__':
    s = input("请输入句子： ")
    input_id = torch.LongTensor(encode(s)).unsqueeze(0)
    # print(model(input_id)[0])
    y_pre = model(input_id)[0]
    y_pre = torch.argmax(torch.softmax(y_pre, dim=-1), dim=-1).squeeze().tolist()
    id2label = load_label_dict(Config)
    res = show(s, y_pre)
    print("".center(100, "="))
    for key in res:
        print("{}类实体：".format(key), end='')
        for value in res[key]:
            print(value, end='    ')
        print("")
    print("".center(100, "="))
