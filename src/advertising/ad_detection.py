#coding: utf-8

import os
import numpy as np
from src.onnx_abs.abs_model import AbsModel

class AdDetection(AbsModel):
    def __init__(self, model_file, dict_path):
        super().__init__(model_file)
        self.word_dict = {}
        self.load_dict(dict_path)

    def load_dict(self, dict_path):
        if not os.path.exists(dict_path):
            raise ValueError("not find file path {}".format(
                dict_path))
        print('load ad dict...')
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split(' ')
                self.word_dict[line[0]] = int(line[1])

    def is_ad(self, sent:str, threshold=0.5) -> bool:
        flag = False
        token_ids = self.parse(sent)
        prob = self.infer(token_ids)
        if prob > threshold:
            flag = True
        return flag

    def parse(self, sent:str, max_len=500):
        sent = list(sent)
        sent = sent[:max_len]
        pad = ['<pad>']*(max_len - len(sent))
        sent += pad
        token_ids = [self.word_dict[token] if token in self.word_dict else 0 for token in sent]
        token_ids = np.array(token_ids, dtype=np.int64)
        token_ids = np.expand_dims(token_ids, 0)
        return token_ids

    def infer(self, token_ids):
        outputs = self.sess.run([self.sess.get_outputs()[0].name], {self.sess.get_inputs()[0].name: token_ids})
        prob = outputs[0]
        y = np.exp(prob[0] - np.max(prob[0]))
        f_x = y / np.sum(np.exp(prob[0]))
        return f_x[1]

if __name__ == '__main__':
    ad_model_path = r'E:\pycharm project\DataFine\weights\advertising\ad_pred.onnx'
    ad_dict_path = r'E:\pycharm project\DataFine\weights\advertising\dict.txt'
    ad_model = AdDetection(ad_model_path, ad_dict_path)
    # print(ad_model.sess.get_inputs()[0])
    # print(ad_model.sess.get_outputs()[0])
    print(ad_model.is_ad('这款洗发水不错'))


