# -*-coding:utf-8-*-
import os

dict_path = r"D:\Text_Categorization_System_TCS\pre_train\data_resource\dict.txt"
tmp_path = r"D:\Text_Categorization_System_TCS\pre_train\data_resource\dict_copy.txt"


def adjust_dict_file(des_path, copy_path):
    with open(des_path, mode='rt', encoding='utf-8') as f, \
            open(copy_path, mode='wt', encoding='utf-8') as f2:
        word_lst = f.readline().strip(' {}').split(",")
        for i in word_lst:
            seg = i.strip().split(":")
            word = seg[0]
            word = word.strip(" '")
            f2.write(word + '\n')
    with open(des_path, mode='wt', encoding='utf-8') as f, \
            open(copy_path, mode='rt', encoding='utf-8') as f2:
        for line in f2:
            if set(line) & set('\'\"'):
                continue
            f.write(line)
    os.remove(copy_path)


if __name__ == '__main__':
    adjust_dict_file(dict_path, tmp_path)
