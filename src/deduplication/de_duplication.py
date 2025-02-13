#coding: utf-8

from simhash import Simhash
import jieba.posseg as pseg
import pickle
import os

class DeDuplication:

    def __init__(self, hash_file):
        self.hash_set = None
        self.hash_file = hash_file
        self.load_hash()

    def haming_distance(self, code_s1, code_s2):
        x = (code_s1 ^ code_s2) & ((1 << 64) - 1)
        ans = 0
        while x:
            ans += 1
            x &= x - 1
        return ans

    def get_similarity(self, a, b):
        if a > b:
            return b / a
        else:
            return a / b

    def get_features(self, string):
        word_list = [word.word for word in pseg.cut(string) if
                     word.flag[0] not in ['u', 'x', 'w', 'o', 'p', 'c', 'm', 'q']]
        return word_list

    def get_distance(self, code_s1, code_s2):
        return self.haming_distance(code_s1, code_s2)

    def get_code(self, string):
        return Simhash(self.get_features(string)).value

    def distance(self, s1:str, s2:str):
        code_s1 = self.get_code(s1)
        code_s2 = self.get_code(s2)
        similarity = (100 - self.haming_distance(code_s1, code_s2) * 100 / 64) / 100
        return similarity

    def is_duplicate(self, s2:str, threshold=0.9) -> bool:
        flag = False
        code_s2 = self.get_code(s2)
        if len(self.hash_set) != 0:
            for code_s1 in self.hash_set:
                similarity = (100 - self.haming_distance(code_s1, code_s2) * 100 / 64) / 100
                if similarity > threshold:
                    flag = True
                    break
        self.hash_set.add(code_s2)
        return flag

    def load_hash(self):
        if os.path.exists(self.hash_file):
            print('load hash file...')
            with open(self.hash_file, 'rb') as f:
                self.hash_set = pickle.load(f)
        else:
            self.hash_set = set()

    def save_hash(self):
        print('save hash file...')
        with open(self.hash_file, 'wb') as f:
            pickle.dump(self.hash_set, f)



if __name__ == '__main__':
    text1 = '我喜欢你'
    text2 = '我讨厌你'
    ded = DeDuplication('hash_set.pkl')
    flag = ded.is_duplicate(text2)
    print(flag)
    ded.save_hash()


