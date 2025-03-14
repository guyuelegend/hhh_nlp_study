#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：BPE_Tokenizition 
@File    ：vocab_generator.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/3/14 9:40 
"""
import os


class TokenTop:
    def __init__(self):
        self.decode_dict = [i for i in range(256)]
        self.encode_dict = {(i,): v for i, v in enumerate(self.decode_dict)}
        self.encode_size = len(self.encode_dict)
        self.vocab = {}

    def encode(self, f_str,vocab_size):
        assert vocab_size > 256
        corpus = "".encode('utf-8')
        with open(f_str, mode='rb') as f:
            line = "  "
            while len(line) != 0:
                line = f.read(1024)
                corpus += line
        corpus = list(corpus)
        print("语料准备完毕！")
        for i in range(256,vocab_size):
            pair_dic = self.get_stats(corpus)
            pair = max(pair_dic,key=lambda x:pair_dic[x])
            corpus = self.merges(corpus,pair,i)
            self.decode_dict.append(pair)
            self.encode_dict[pair] = i
            self.encode_size += 1
        print("字典准备完毕")
        for k ,v in self.encode_dict.items():
            print(f"{k}: {v}")
        for k,v in self.encode_dict.items():
            self.vocab[self.decode(k)] = v
        print(self.vocab)

    def decode(self,tokens):
        for i in range(self.encode_size - 1, 255, -1):
            max_pair = self.decode_dict[i]
            tokens = self.emerges(tokens, max_pair, i)
        # print(tokens)
        tokens = b"".join(bytes([idx]) for idx in tokens)
        tokens = tokens.decode("utf-8", errors="replace")
        return tokens



    @staticmethod
    def get_stats(ids):
        """
        在utf-8 编码后统计二元词组的次数
        """
        counts = {}
        for pair in zip(ids[:-1], ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    @staticmethod
    def merges(ids, pair, idx):
        """根据二元词组来替换原 utf-8 编码"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    @staticmethod
    def emerges(ids, pair, idx):
        """根据二元词组字典来还原原 utf-8 编码"""
        new_ids = []
        for i in ids:
            if i == idx:
                one, two = pair
                new_ids.append(one)
                new_ids.append(two)
            else:
                new_ids.append(i)
        return new_ids


BASE_PATH = r"D:\BPE_Tokenizition\Heroes"
file_lst = os.listdir(BASE_PATH)
total_path = os.path.join(BASE_PATH, "total.txt")
if "total.txt" in file_lst:
    file_lst.remove("total.txt")
f_out = open(total_path, mode='wb')
for f_name in file_lst:
    path_tmp = os.path.join(BASE_PATH, f_name)
    with open(path_tmp, mode='rb') as f_in:
        for line in f_in:
            f_out.write(line)
        f_out.write('\n'.encode('utf-8'))
print(f_out.tell())
f_out.close()
if __name__ == '__main__':
    obj = TokenTop()
    obj.encode(total_path,2000)