# coding:utf-8
"""
我们将训练好的词向量模型来组成句子向量，然后采用Kmeans对句向量进行分类，打标签
"""
import jieba
import gensim
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import math
import re
import json
import time

# 调整logger日志的输出等级
# jieba.logger.setLevel(20)
root_path = r"D:\NLP2025\深度学习\Dataset"
model_path = r"{}\word_vec_model.w2v".format(root_path)
corpus_path = r"{}\Train.txt".format(root_path)


class MyCorpus:
    def __init__(self, start, end, corpus_path):
        self.start = start
        self.end = end
        self.dataf = open(corpus_path, mode="rt", encoding="utf-8")
        self.data = self.dataf.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        line = self.data.__next__()
        if self.start < self.end:
            tmp = line.strip()[5:].strip()
            res = " ".join(list(jieba.cut(tmp)))
            self.start += 1
            return res
        else:
            raise StopIteration

    def __del__(self):
        self.dataf.close()


def sentences_to_vectors(sentences, model):
    """
    主要逻辑就是加载模型中的词向量，将一句话中的词向量按列相加，然后这句话的总词数求平均
    得到每个句子的向量，然后句子向量numpy矩阵返回
    """
    vectors = []
    for i in sentences:
        word_seg = i.split(" ")
        vector = np.zeros(model.vector_size)
        for word in word_seg:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(word_seg))
    return np.array(vectors)


def main():
    """
    1. 加载训练好模型的词向量空间
    2. 加载分词后的句子
    3. 将分词后的句子转化成句向量
    4. 指定Kmeans的分类别数，并调用fit方法训练分类
    5. 将同样标签的句子添加到一起
    6. 打印部分结果展示内容
    """
    model = Word2Vec.load(model_path)
    sentances = MyCorpus(0, 52476, corpus_path)
    vectors = sentences_to_vectors(sentances, model)

    n = int(math.sqrt(52476))
    print("通常我们聚类的数量为总数据开根号先试试效果：", n)
    kmeans = KMeans(n)
    kmeans.fit(vectors)
    print("kmeans聚类结束")
    # 必须再次开一个迭代器，上一个迭代器已经取完元素，无法使用
    sentances_label = defaultdict(list)
    sentances = MyCorpus(0, 52476, corpus_path)

    """
    1. 我们获取了每个聚类点下的所有的标签，首先转化成对应的句向量
    2. 计算每个标签下句向量之间的距离，这里我们采用cos向量夹角来算，然后求他们余弦值平均值
    3. 按照类内平均距离进行排序，选择前面几项进行展示
    """
    # print(len(kmeans.cluster_centers_[-1]))
    count = 1
    ave_value = {}
    for sentance, label in zip(sentances, kmeans.labels_):
        sentances_label[label].append(sentance)
    for label, sentance_lst in sentances_label.items():
        sentances_label_vector = sentences_to_vectors(sentance_lst, model)
        label_index = int(label)
        center_vector = kmeans.cluster_centers_[label_index]
        length = 0
        ave_value[label] = np.float32(0)
        for i in sentances_label_vector:
            up_num = np.dot(center_vector, i)
            down_num = np.sqrt(np.sum(np.square(i)))
            ave_value[label] += up_num / down_num
            length += 1
        if count == 1:
            print(ave_value.items())
            count += 1
        print("距离和：", ave_value[label], "    该类句子个数：", length)
        ave_value[label] /= np.float32(length)
    ans = [(labels, ave_value) for labels, ave_value in ave_value.items()]
    sorted(ans, key=lambda x: x[1], reverse=True)
    print("排序完成")
    show_num = 20
    print("我们选择前{}个平均距离较小的类别进行展示".format(show_num))
    for i in range(show_num):
        label, ave_value = ans[i]
        print("标签：", label, "   ", "该类类内平均距离：", ave_value)

    # for label, sentancess in sentances_label.items():
    #     print("cluster {}".format(label), int(label), type(int(label)))
    #     for i in range(min(5, len(sentancess))):
    #         print((sentancess[i].replace(" ", "")))
    #     print("".center(50, "-"))


if __name__ == '__main__':
    main()
    # s1 = np.sum(np.multiply(np.array([1, 2, 3]), np.array([8, 9, 7])))
    # s2 = np.dot(np.array([1, 2, 3]), np.array([8, 9, 7]))
    # print(float(s1), type(s1))
    # print(float(s2), type(s2))
    # print(float(s2 / s1), type(s2 / s1))
