# coding:utf-8
import torch
from transformers import BertModel
import math
import numpy as np

PRE_TRAIN_PATH = r"E:\关于视频学习资料\NLP_八斗教学\第六周 语言模型\bert-base-chinese"

bert = BertModel.from_pretrained(PRE_TRAIN_PATH, return_dict=False)
state_dict = bert.state_dict()
bert.eval()
# x = np.array([2450, 15486, 102, 2110])
# torch_x = torch.LongTensor([x])
# seqence_output, pooler_output = bert(torch_x)
# print("seqence_output:\n",seqence_output,
#       "\npooler_output:\n",pooler_output)
# print(bert.state_dict().keys())
# embedding+tranformer*12的所有可训练参数统计

total = 0
tmp = 0
vocab_size = bert.state_dict()["embeddings.word_embeddings.weight"].size()[0]
word_dim = bert.state_dict()["embeddings.word_embeddings.weight"].size()[1]
total += vocab_size * word_dim + 2 * word_dim + 512 * word_dim + 1 * word_dim + 1 * word_dim + (word_dim * word_dim * 3 + \
            3 * word_dim + 1 * word_dim + 1 * word_dim + word_dim * 4 * word_dim + 4 * word_dim + 4 * word_dim * word_dim + \
            1 * word_dim + 1 * word_dim + 1 * word_dim + word_dim * word_dim + 1 * word_dim) * 12
tmp += word_dim * word_dim * 3 + \
            3 * word_dim + 1 * word_dim + 1 * word_dim + word_dim * 4 * word_dim + 4 * word_dim + 4 * word_dim * word_dim + \
            1 * word_dim + 1 * word_dim + 1 * word_dim + word_dim * word_dim + 1 * word_dim
print("单层transformer中可训练参数总数：", tmp)
print("embedding层+12层transformer中可训练参数总数：", total)

total = 0
# embedding层
total += bert.state_dict()["embeddings.word_embeddings.weight"].size()[0]*bert.state_dict()["embeddings.word_embeddings.weight"].size()[1]#vocab_size*word_dim
total += bert.state_dict()["embeddings.token_type_embeddings.weight"].size()[0] * bert.state_dict()["embeddings.token_type_embeddings.weight"].size()[1]#2*word_dim
total += bert.state_dict()["embeddings.position_embeddings.weight"].size()[0] * bert.state_dict()["embeddings.position_embeddings.weight"].size()[1]#512*word_dim
total += bert.state_dict()["embeddings.LayerNorm.weight"].size()[-1] + bert.state_dict()["embeddings.LayerNorm.bias"].size()[-1]#1*word_dim + 1*word_dim
print("embedding层中可训练参数总数：", total)
tmp = 0
tmp += bert.state_dict()["encoder.layer.0.attention.self.query.weight"].size()[0] * bert.state_dict()["encoder.layer.0.attention.self.query.weight"].size()[1]
tmp += bert.state_dict()["encoder.layer.0.attention.self.query.bias"].size()[-1]
tmp += bert.state_dict()["encoder.layer.0.attention.self.key.weight"].size()[0] * bert.state_dict()["encoder.layer.0.attention.self.key.weight"].size()[1]
tmp += bert.state_dict()["encoder.layer.0.attention.self.key.bias"].size()[-1]
tmp += bert.state_dict()["encoder.layer.0.attention.self.value.weight"].size()[0] * bert.state_dict()["encoder.layer.0.attention.self.value.weight"].size()[1]
tmp += bert.state_dict()["encoder.layer.0.attention.self.value.bias"].size()[-1]
tmp += bert.state_dict()["encoder.layer.0.attention.output.LayerNorm.weight"].size()[-1] + bert.state_dict()["encoder.layer.0.attention.output.LayerNorm.bias"].size()[-1]
tmp += bert.state_dict()["encoder.layer.0.intermediate.dense.weight"].size()[0] * bert.state_dict()["encoder.layer.0.intermediate.dense.weight"].size()[1]
tmp += bert.state_dict()["encoder.layer.0.intermediate.dense.bias"].size()[-1]
tmp += bert.state_dict()["encoder.layer.0.output.dense.weight"].size()[0] * bert.state_dict()["encoder.layer.0.output.dense.weight"].size()[1]
tmp += bert.state_dict()["encoder.layer.0.output.dense.bias"].size()[-1]
tmp += bert.state_dict()["encoder.layer.0.output.LayerNorm.weight"].size()[-1] + bert.state_dict()["encoder.layer.0.output.LayerNorm.bias"].size()[-1]
tmp += bert.state_dict()["pooler.dense.weight"].size()[0] * bert.state_dict()["pooler.dense.weight"].size()[1]
tmp += bert.state_dict()["pooler.dense.bias"].size()[-1]
print("单层transformer中可训练参数总数：", tmp)
print("embedding层+12层transformer中可训练参数总数：", total + 12 * tmp)

"""
embedding层中可训练参数总数： 16622592
单层transformer中可训练参数总数： 7087872
embedding层+12层transformer中可训练参数总数： 101677056
"""
"""
# embedding层   ==>input:sentence_size*vocab_size
print(bert.state_dict()["embeddings.word_embeddings.weight"].size())#vocab_size*word_dim
print(bert.state_dict()["embeddings.token_type_embeddings.weight"].size())#2*word_dim
print(bert.state_dict()["embeddings.position_embeddings.weight"].size())#512*word_dim
# layer_normalize
print(bert.state_dict()["embeddings.LayerNorm.weight"].size())#1*word_dim
print(bert.state_dict()["embeddings.LayerNorm.bias"].size())#1*word_dim
# Q K V
print(bert.state_dict()["encoder.layer.0.attention.self.query.weight"].size())#word_dim*word_dim
print(bert.state_dict()["encoder.layer.0.attention.self.query.bias"].size())#1*word_dim
print(bert.state_dict()["encoder.layer.0.attention.self.key.weight"].size())#word_dim*word_dim
print(bert.state_dict()["encoder.layer.0.attention.self.key.bias"].size())#1*word_dim
print(bert.state_dict()["encoder.layer.0.attention.self.value.weight"].size())#word_dim*word_dim
print(bert.state_dict()["encoder.layer.0.attention.self.value.bias"].size())#1*word_dim
# layer_normalize
print(bert.state_dict()["encoder.layer.0.attention.output.LayerNorm.weight"].size())#1*word_dim
print(bert.state_dict()["encoder.layer.0.attention.output.LayerNorm.bias"].size())#1*word_dim
# feed forward
print(bert.state_dict()["encoder.layer.0.intermediate.dense.weight"].size())#word_dim*  4*word_dim
print(bert.state_dict()["encoder.layer.0.intermediate.dense.bias"].size())#4*word_dim
print(bert.state_dict()["encoder.layer.0.output.dense.weight"].size())#4*word_dim   * word_dim
print(bert.state_dict()["encoder.layer.0.output.dense.bias"].size())#1*word_dim
# layer_normalize
print(bert.state_dict()["encoder.layer.0.output.LayerNorm.weight"].size())#1*word_dim
print(bert.state_dict()["encoder.layer.0.output.LayerNorm.bias"].size())#1*word_dim
# pooler
print(bert.state_dict()["pooler.dense.weight"].size())#word_dim*word_dim
print(bert.state_dict()["pooler.dense.bias"].size())#1*word_dim
"""
