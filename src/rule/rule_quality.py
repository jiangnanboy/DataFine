#coding: utf-8
import re
import jieba

class RuleFilter:
    def is_english(self, text:str) -> bool:
        flag = True
        text = list(text)
        for t in text:
            if self.is_chinese(t):
                flag = False
                break
        return flag

    def is_chinese(self, text:str) -> bool:
        return '\u4e00' <= text <= '\u9fff'

    def is_blank(self, text:str) -> bool:
        text = text.strip()
        return len(text) == 0

    def is_all_digial(self, text:str) -> bool:
        regex = r'^\d+$'
        return self.is_match(regex, text)

    def is_lessthan_size(self, text:str, size=10) -> bool:
        text = list(text)
        return len(text) <= size

    def is_lessthan_percent(self, text:str, percent=0.5) -> bool:
        flag = True
        regex = r'[\u4e00-\u9fff]+'
        chinese_text = self.find_all_match(regex, text)
        chinese_text = ''.join(chinese_text)
        if (1.0 * len(chinese_text) / len(text)) >= percent:
            flag = False
        return flag

    def is_lessthan_percent_chinese_english_number(self, text:str, percent=0.4) -> bool:
        flag = True
        regex_chinese = r'[\u4e00-\u9fff]+'
        regex_english = r'[a-zA-Z]'
        regex_number = r'\d+'
        chinese_text = self.find_all_match(regex_chinese, text)
        english_text = self.find_all_match(regex_english, text)
        number_text = self.find_all_match(regex_number, text)
        chinese_text = ''.join(chinese_text)
        english_text = ''.join(english_text)
        number_text = ''.join(number_text)
        if (1.0 * (len(chinese_text) + len(english_text) + len(number_text)) / len(text)) >= percent:
            flag = False
        return flag

    def is_much_or_little_punctuation(self, text:str, little_percent=0.005, much_percent=0.5) -> bool:
        flag = True
        punctuation = r"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~。？！，、；：：”“（）——……《 》▪—"
        punctuation = list(punctuation)
        text = list(text)
        punctuation_count = 0
        for t in text:
            if t in punctuation:
                punctuation_count += 1
        if ((1.0 * punctuation_count / len(text) >= little_percent) or (
                1.0 * punctuation_count / len(text) <= much_percent)):
            flag = False
        return flag

    def is_last_token_punctuation(self, text:str, punctuation=["。", ".", "!", "！", "?", "？", "……", "…"]) -> bool:
        flag = True
        last_token = text[len(text) - 1].strip()
        if last_token in punctuation:
            flag = False
        return flag

    def is_id_phone_email_url_ip(self, text:str):
        flag = False
        url_regex = r"(http(s)?://)?([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?"
        email_regex = r"\\w+@\\w+\\.[a-z]+(\\.[a-z]+)?"
        id_regex = r"\\w+@\\w+\\.[a-z]+(\\.[a-z]+)?"
        mobile_regex = r"(\\+\\d+)?1[3458]\\d{9}$"
        telephone_regex = r"(\\+\\d+)?(\\d{3,4}\\-?)?\\d{7,8}$"
        ip_regex = r"\\d+.\\d+.\\d+.\\d+"
        if (self.is_match(url_regex, text) or self.is_match(email_regex, text) or self.is_match(id_regex, text) or
                self.is_match(mobile_regex, text) or self.is_match(telephone_regex, text) or self.is_match(ip_regex, text)):
            flag = True
        return flag

    def is_less_text_density(self, text:str, percent=0.6) -> bool:
        flag = False
        punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~。？！，、；：：”“（）——……《 》▪—"
        punctuation = list(punctuation)
        word_list = list(jieba.cut(text))
        word_list = [word for word in word_list if word not in punctuation]
        word_list_count = len(word_list)
        word_list = [word for word in word_list if len(word) >= 2]
        word_list_than_two_count = len(word_list)
        if (1.0 * word_list_than_two_count / word_list_count < percent):
            flag = True
        return flag

    def rule_clean(self, text:str) -> bool:
        flag = self.is_blank(text) or self.is_lessthan_size(text) or self.is_lessthan_percent_chinese_english_number(text) \
        or self.is_lessthan_percent(text) or self.is_much_or_little_punctuation(text) or self.is_last_token_punctuation(text) \
        or self.is_id_phone_email_url_ip(text) or self.is_all_digial(text) or self.is_less_text_density(text)
        return flag

    def consecutive_special_symbol(self, text:str) -> str:
        consecutive_regex = r"[-]{4,}|[.]{4,}|[=]{4,}"
        return self.match_replacement(consecutive_regex, text)

    def special_symbol_no_meaning(self, text:str) -> str:
        special_regex = r"[\¤\⌒\々\〓\▌\◇\▲\△\▪\★\◆\▼\●\▽\◁\☆\○]"
        return self.match_replacement(special_regex, text)

    def rule_special_symbol(self, text:str) -> str:
        text = self.consecutive_special_symbol(text)
        text = self.special_symbol_no_meaning(text)
        return text

    def find_all_match(self, regex:str, text:str) -> str:
        pattern = re.compile(regex)
        result_text = pattern.findall(text)
        return result_text

    def is_match(self, regex:str, text:str) -> bool:
        return bool(re.match(regex, text))

    def is_in_regex(self, w, regex) -> bool:
        return w in regex

    def match_replacement(self, regex:str, text:str) -> str:
        pattern = re.compile(regex)
        return re.sub(pattern, '', text)

