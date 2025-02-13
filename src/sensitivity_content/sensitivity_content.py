#coding: utf-8
import numpy as np
from src.onnx_abs.abs_model import AbsModel
from src.sensitivity_content.tokenizer import SentTokenizer


class SensitivityContentDetection(AbsModel):
    def __init__(self, model_file, sent_tokenizer):
        super().__init__(model_file)
        self.sent_tokenizer = sent_tokenizer

    def is_sen_content(self, sent: str) -> bool:
        flag = False
        token_ids = self.parse(sent)
        probs = self.infer(token_ids)
        probs = probs.tolist()
        max_prob = float('-inf')
        label = 0
        for i in range(len(probs)):
            if probs[i] > max_prob:
                max_prob = probs[i]
                label = i
        if label == 1:
            flag = True
        return flag

    def parse(self, sent: str, max_len=512):
        sent_tokenizer_ids = self.sent_tokenizer.tokenizer(sent, truncation=True, max_length=max_len)
        return sent_tokenizer_ids

    def infer(self, token_ids):
        input_ids = np.array(token_ids['input_ids'], dtype=np.int64)
        token_type_ids = np.array(token_ids['token_type_ids'], dtype=np.int64)
        attention_mask = np.array(token_ids['attention_mask'], dtype=np.int64)
        input_ids = np.expand_dims(input_ids, 0)
        token_type_ids = np.expand_dims(token_type_ids, 0)
        attention_mask = np.expand_dims(attention_mask, 0)
        outputs = self.sess.run([self.sess.get_outputs()[0].name], {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask})
        prob = outputs[0]
        y = np.exp(prob[0] - np.max(prob[0]))
        f_x = y / np.sum(np.exp(prob[0]))
        return f_x


if __name__ == '__main__':
    se_model_path = r'Z:\project\content_det\roberta_insult_onnx\roberta_wwm_insult_model.onnx'
    se_vocab_path = r'E:\pycharm project\DataFine\weights\security\vocab'
    sent_tokenizer = SentTokenizer(se_vocab_path)

    se_model = SensitivityContentDetection(se_model_path, sent_tokenizer)
    print(se_model.sess.get_inputs()[0])
    print(se_model.sess.get_outputs()[0])
    print(se_model.is_sen_content('黑人很多都好吃懒做，偷奸耍滑！'))
