# -*-coding:utf-8-*-
from core.data_loader import data_load
import torch


class Evaluate:
    def __init__(self, config, model):
        self.config = config
        self.valid_data = data_load(config, config["valid_data_path"])
        self.train_data = data_load(config, config["train_data_path"])
        self.state_dict = {"wrong": 0, "correct": 0}
        self.model = model
        self.question_dict = {}
        self.question_id_set = []

    def knwb_to_vector(self):
        for standard_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_dict[len(self.question_id_set)] = standard_index
                self.question_id_set.append(question_id)
        question_matrixs = torch.stack(self.question_id_set, dim=0)
        with torch.no_grad():
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)

    def eval(self):
        self.state_dict = {"wrong": 0, "correct": 0}
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                test_questions = self.model(input_ids)
            self.write_state(test_questions, labels)
        acc = self.show_state()
        return acc

    def write_state(self, test_questions, labels):
        assert len(test_questions) == len(labels)
        test_questions = torch.nn.functional.normalize(test_questions, dim=-1)
        for test_question, label in zip(test_questions, labels):
            res = torch.mm(test_question.unsqueeze(0), self.knwb_vectors.T)
            question_index = int(torch.argmax(res.squeeze()))
            question_index = self.question_dict[question_index]
            if question_index == int(label):
                self.state_dict["correct"] += 1
            else:
                self.state_dict["wrong"] += 1
        return

    def show_state(self):
        correct_num = self.state_dict["correct"]
        wrong_num = self.state_dict["wrong"]
        print("".center(100, "="))
        print("正确的数目：{}".format(correct_num))
        print("错误的数目：{}".format(wrong_num))
        print("正确率：{:.2%}".format(correct_num / (correct_num + wrong_num)))
        print("".center(100, "="))
        return correct_num / (correct_num + wrong_num)


if __name__ == '__main__':
    print("{:.2%}".format(0.123456))
