# -*-coding:utf-8-*-
from core.data_loader import load_data
import torch


class Evaluator:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.valid_data = load_data(config, self.config["valid_data_path"])
        self.collector = {"correct": 0, "wrong": 0}

    def load(self, epoch):
        print("start".center(100, "-"))
        print("第{}次测试开始......".format(epoch))
        # 模型前向计算
        self.model.eval()
        self.collector = {"correct": 0, "wrong": 0}
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            x_data, y_data = batch_data
            with torch.no_grad():
                y_predicate = self.model(x_data)
            y_predicate = torch.argmax(y_predicate, dim=-1)
            assert len(y_predicate) == len(y_data)
            # if not len(y_predicate) == len(y_data):
            #     raise AssertionError
            for label, predicate in zip(y_data, y_predicate):
                if int(label) == int(predicate):
                    self.collector["correct"] += 1
                else:
                    self.collector["wrong"] += 1
        acc = self.show_state()
        print("end".center(100, "-"))
        return acc

    def show_state(self):
        correct_num = self.collector["correct"]
        wrong_num = self.collector["wrong"]
        correct_rate = correct_num / (correct_num + wrong_num)
        print("测试总样本个数{}，\n".format(correct_num + wrong_num),
              "预测正确的个数：{},预测正确的个数：{}\n".format(correct_num, wrong_num),
              "此次正确率为{:.2%}".format(correct_rate), sep='')
        return correct_rate
