# -*-coding:utf-8-*-

import os
import json
import random
from collections import defaultdict

random.seed(678)
train_data_path = r"D:\Text_Categorization_System_TCS\pre_train\data_resource\Train.txt"
valid_data_path = r"D:\Text_Categorization_System_TCS\pre_train\data_resource\Valid_seg.txt"
title_dict_path = r"D:\Text_Categorization_System_TCS\pre_train\data_resource\title_set.json"
train_copy_data_path = r"D:\Text_Categorization_System_TCS\pre_train\data_resource\Train_seg.txt"

seg_percentage = 0.8
valid_num_by_title = 800
train_num_by_title = 2420


def title_dict_generator(train_path, title_path):
    with open(train_path, mode='rt', encoding='utf-8') as f, \
            open(title_path, mode='wt', encoding='utf-8') as f2:
        title_dict = dict()
        title_num = defaultdict(int)
        index = 0

        for line in f:
            title = line[:5].strip()[-2:]
            if title_dict.get(title) is None:
                title_dict[title] = index
                index += 1
            title_num[title] += 1
        json.dump(title_dict, f2, ensure_ascii=False)
        print(title_num)
        return title_num


def split_valid(train_path, valid_path, train_copy_path, title_num):
    with open(train_path, mode='rt', encoding='utf-8') as ft, \
            open(train_copy_path, mode='wt', encoding='utf-8') as fc, \
            open(valid_path, mode='wt', encoding='utf-8') as fv:
        count = 0
        count2 = 0
        for line in ft:
            title = line[:5].strip()[-2:]
            if title_num[title] != 0 and count < valid_num_by_title:
                fv.write(line)
                count += 1
            elif title_num[title] != 0 and count2 < train_num_by_title:
                fc.write(line)
                count2 += 1
            title_num[title] -= 1
            if title_num[title] == 0:
                count = 0
                count2 = 0


if __name__ == '__main__':
    # title_num = title_dict_generator(train_data_path, title_dict_path)
    # split_valid(train_data_path,valid_data_path,train_copy_data_path,title_num)

    title_dict_generator(valid_data_path, title_dict_path)
    title_dict_generator(train_copy_data_path, title_dict_path)

"""
{
'财经': 33389, 
'彩票': 6830, 
'房产': 18045, 
'股票': 138959, 
'家居': 29328, 
'教育': 37743, 
'科技': 146637, 
'社会': 45765, 
'时尚': 12032, 
'时政': 56778, 
'体育': 118444, 
'星座': 3221, 
'游戏': 21936, 
'娱乐': 83369})
"""
