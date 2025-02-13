
from transformers import BertTokenizer

class SentTokenizer:
    def __init__(self, vocab_path):
        self.tokenizer = self.load_tokenizer(vocab_path)

    def load_tokenizer(self, vocab_path):
        print('load tokenizer...')
        tokenizer = BertTokenizer.from_pretrained(vocab_path)
        return tokenizer