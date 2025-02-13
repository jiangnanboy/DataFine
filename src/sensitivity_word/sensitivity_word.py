#coding: utf-8
import os

class SensitivityWordDetection:
    def __init__(self, dict_path):
        self.keyword_chains = {}
        self.delimit = '\x00'
        self.load_dict(dict_path)

    def load_dict(self, dict_path):
        if not os.path.exists(dict_path):
            raise ValueError("not find file path {}".format(
                dict_path))
        print('load sensitivity dict...')
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                self.add(word)

    def add(self, keyword):
        keyword = keyword.lower()
        chars = keyword.strip()
        if not chars:
            return
        level = self.keyword_chains
        for i in range(len(chars)):
            if chars[i] in level:
                level = level[chars[i]]
            else:
                if not isinstance(level, dict):
                    break
                for j in range(i, len(chars)):
                    level[chars[j]] = {}
                    last_level, last_char = level, chars[j]
                    level = level[chars[j]]
                last_level[last_char] = {self.delimit: 0}
                break
        if i == len(chars) - 1:
            level[self.delimit] = 0

    def filter(self, sent:str, repl="*") -> str:
        sent = sent.lower()
        ret = []
        start = 0
        while start < len(sent):
            level = self.keyword_chains
            step_ins = 0
            for char in sent[start:]:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        ret.append(repl * step_ins)
                        start += step_ins - 1
                        break
                else:
                    ret.append(sent[start])
                    break
            else:
                ret.append(sent[start])
            start += 1

        return ''.join(ret)

    def is_contain_sensitivity(self, sent:str) -> bool:
        flag = False
        sent = sent.lower()
        start = 0
        while start < len(sent):
            level = self.keyword_chains
            step_ins = 0
            for char in sent[start:]:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        flag = True
                        start += step_ins - 1
                        break
                else:
                    break
            start += 1

        return flag


if __name__ == "__main__":
    sensitivity_word_path = r'/weights/sensitivity_word\sensi_words.txt'
    sen_det = SensitivityWordDetection(sensitivity_word_path)

    import time
    t = time.time()
    print(sen_det.is_contain_sensitivity("法轮功 我操操操"))
    print(sen_det.is_contain_sensitivity("针孔摄像机 我操操操"))
    print(sen_det.is_contain_sensitivity("售假人民币 我操操操"))
    print(sen_det.is_contain_sensitivity("传世私服 我操操操"))
    print(time.time() - t)

