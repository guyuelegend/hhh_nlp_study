# -*-coding:utf-8-*-
import torch
from torch import nn
from transformers import BertModel
from torch.optim import SGD, Adam


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        vocab_size = config["vocab_size"] + 3
        hidden_size = config["hidden_size"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.used_bert = False
        if self.config["model_type"] == "fast_text":
            self.encoder = lambda x: x
        elif self.config["model_type"] == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=config["hidden_layer_num"], batch_first=True, )
        elif self.config["model_type"] == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=config["hidden_layer_num"], batch_first=True, )
        elif self.config["model_type"] == "grn":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=config["hidden_layer_num"], batch_first=True, )
        elif self.config["model_type"] == "cnn":
            self.encoder = CNN(config)
        elif self.config["model_type"] == "gate_cnn":
            self.encoder = GateCNN(config)
        elif self.config["model_type"] == "stack_gate_cnn":
            self.encoder = StackGateCNN(config)
        elif self.config["model_type"] == "rcnn":
            self.encoder = RCNN(config)
        elif self.config["model_type"] == "bert":
            self.encoder = BertModel.from_pretrained(self.config["bert_pre_train_path"], return_dict=False)
            self.used_bert = True
            hidden_size = self.encoder.config.hidden_size
        elif self.config["model_type"] == "bert_cnn":
            self.encoder = BertCNN(config)
            self.used_bert = True
            hidden_size = self.encoder.bert.config.hidden_size
        elif self.config["model_type"] == "bert_lstm":
            self.encoder = BertLSTM(config)
            self.used_bert = True
            hidden_size = self.encoder.bert.config.hidden_size
        elif self.config["model_type"] == "bert_mid_layer":
            self.encoder = BertMidLayer(config)
            self.used_bert = True
            hidden_size = self.encoder.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size, config["classify_num"])
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None):
        # batch_size*max_seq*vocab_size ==> batch_size*max_seq*hidden_size
        # batch_size*max_seq*hidden_size ==> batch_size*max_seq*hidden_size
        if self.used_bert is True:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        if isinstance(x, tuple):
            x, _ = x
        # batch_size*max_seq*hidden_size ==> batch_size*1*hidden_size
        if self.config["pooling_type"] == "max":
            self.pooling = nn.MaxPool1d(x.shape[1])
        elif self.config["pooling_type"] == 'cls':
            self.pooling = lambda x: x[:, :, 0]
        else:
            # self.config["pooling_type"] == "avg":
            self.pooling = nn.AvgPool1d(x.shape[1])
        # batch_size*1*hidden_size ==> batch_size*hidden_size
        x = self.pooling(x.transpose(1, 2)).squeeze()
        # batch_size*hidden_size ==> batch_size*classify_num
        y_predicate = self.classify(x)
        if y is not None:
            return self.loss_func(y_predicate, y.squeeze())
        else:
            return y_predicate


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        # padding是因为张量卷积之后会发生形状的变形，所以padding是补充张量左右两边向量（零），是其恢复到输入x的形状
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):
        # batch_size*max_seq*hidden_size ==> batch_size*max_seq*hidden_size有padding，卷积后形状不发生变化
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        return x


class GateCNN(nn.Module):
    def __init__(self, config):
        super(GateCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        # batch_size*max_seq*hidden_size ==> batch_size*max_seq*hidden_size
        gate = self.activate(self.gate(x))
        x = self.cnn(x)
        # 点乘不改变矩阵形状
        return torch.mul(x, gate)


class StackGateCNN(nn.Module):
    def __init__(self, config):
        super(StackGateCNN, self).__init__()
        self.num_layer = config["hidden_layer_num"]
        self.hidden_size = config["hidden_size"]
        self.gcnn_layers = nn.ModuleList(
            GateCNN(config) for i in range(self.num_layer)
        )
        self.liner_layer1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layer)
        )
        self.liner_layer2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layer)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layer)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layer)
        )

    def forward(self, x):
        # batch_size*max_seq*hidden_size ==> batch_size*max_seq*hidden_size
        for i in range(self.num_layer):
            # 仿照bert的self-attention层少了liner
            gcnn_x = self.gcnn_layers[i](x)
            x = self.bn_after_gcnn[i](x + gcnn_x)
            # 仿照bert的forward_feed_net,将gelu换成relu
            ff = self.liner_layer1[i](x)
            ff = nn.functional.relu(x)
            ff = self.liner_layer2[i](x)
            x = self.bn_after_ff[i](x + ff)
        return x


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.cnn = GateCNN(config)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True, num_layers=config["hidden_layer_num"])

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x


class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_pre_train_path"], return_dict=False)
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x, _ = self.bert(x)
        x = self.cnn(x)
        return x


class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_pre_train_path"], return_dict=False)
        config["hidden_size"] = self.bert.config.hidden_size
        self.rnn = nn.LSTM(config["hidden_size"], config["hidden_size"], batch_first=True)

    def forward(self, x):
        x, _ = self.bert(x)
        x, _ = self.rnn(x)
        return x


class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_pre_train_path"], return_dict=False)
        self.bert.config.output_hidden_states = True
        config["hidden_size"] = self.bert.config.hidden_size

    def forward(self, x):
        x = self.bert(x)
        # print(x[0],x[0].shape)#torch.Size([128, 30, 768])
        # print(x[1],x[1].shape)#torch.Size([128, 768])
        # print(len(x[2]),x[2][-1].shape)#5*torch.Size([128, 30, 768])
        x = x[2]
        x = torch.add(x[-2], x[-1])
        return x


def choose_optim(config, models):
    if config["optim_type"] == 'adam':
        return Adam(models.parameters(), lr=config["learn_rate"])
    else:
        return SGD(models.parameters(), lr=config["learn_rate"])


if __name__ == '__main__':
    from conf.setting import Config
    from core import data_loader

    dg = data_loader.load_data(Config, Config["valid_data_path"]).__iter__()
    input_seq, label_seq = next(dg)
    model = TorchModel(Config)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    # print(choose_optim(Config, model))
    with torch.no_grad():
        if torch.cuda.is_available():
            input_seq = input_seq.cuda()
        y_predicate = model(input_seq)
        # print(torch.argmax(y_predicate, dim=-1, keepdim=True))
        print(torch.argmax(y_predicate, dim=-1))
        print(label_seq.transpose(0, 1).squeeze())
