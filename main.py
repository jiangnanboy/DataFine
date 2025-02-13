from src.rule.rule_quality import RuleFilter
from src.advertising.ad_detection import AdDetection
from src.deduplication.de_duplication import DeDuplication
from src.sensitivity_word.sensitivity_word import SensitivityWordDetection
from src.sensitivity_content.sensitivity_content import SensitivityContentDetection
from src.sensitivity_content.tokenizer import SentTokenizer


class ContentAudit:
    def __init__(self, ruleFilter, adDetection, deDuplication, sensitivityWordDetection,
                 sensitivityContentDetectionInsult, sensitivityContentDetectionPolitic, sensitivityContentDetectionPorn, sensitivityContentDetectionViolence):
        self.ruleFilter = ruleFilter
        self.adDetection = adDetection
        self.deDuplication = deDuplication
        self.sensitivityWordDetection = sensitivityWordDetection
        self.sensitivityContentDetectionInsult = sensitivityContentDetectionInsult
        self.sensitivityContentDetectionPolitic = sensitivityContentDetectionPolitic
        self.sensitivityContentDetectionPorn = sensitivityContentDetectionPorn
        self.sensitivityContentDetectionViolence = sensitivityContentDetectionViolence

    def det(self, sent:str) -> bool:
        bad_flag = False
        if self.ruleFilter.rule_clean(sent):
            bad_flag = True
        if not bad_flag:
            sent = self.ruleFilter.rule_special_symbol(sent)
        if not bad_flag:
            bad_flag = self.adDetection.is_ad(sent)
        if not bad_flag:
            bad_flag = self.deDuplication.is_duplicate(sent)
        if not bad_flag:
            bad_flag = self.sensitivityWordDetection.is_contain_sensitivity(sent)
        if not bad_flag:
            bad_flag = self.sensitivityContentDetectionInsult.is_sen_content(sent)
        if not bad_flag:
            bad_flag = self.sensitivityContentDetectionPolitic.is_sen_content(sent)
        if not bad_flag:
            bad_flag = self.sensitivityContentDetectionPorn.is_sen_content(sent)
        if not bad_flag:
            bad_flag = self.sensitivityContentDetectionViolence.is_sen_content(sent)
        return bad_flag, sent

if __name__ == '__main__':
    # 1 rule filter
    ruleFilter = RuleFilter()

    # 2 advertising
    ad_model_path = r'E:\pycharm project\DataFine\weights\advertising\ad_pred.onnx'
    ad_dict_path = r'E:\pycharm project\DataFine\weights\advertising\dict.txt'
    adDetection = AdDetection(ad_model_path, ad_dict_path)

    # 3 deduplication
    deDuplication = DeDuplication('hash_set.pkl')

    # 4 sensitivity word
    sensitivity_word_path = r'/weights/sensitivity_word/sensi_words.txt'
    sensitivityWordDetection = SensitivityWordDetection(sensitivity_word_path)

    # 5 tokenizer
    se_vocab_path = r'E:\pycharm project\DataFine\weights\security\vocab'
    sent_tokenizer = SentTokenizer(se_vocab_path)

    # 6 insult model
    insult_model_path = 'roberta_wwm_insult_model.onnx'
    sensitivityContentDetectionInsult = SensitivityContentDetection(insult_model_path, sent_tokenizer)

    # 7 politic model
    politic_model_path = 'roberta_wwm_insult_model.onnx'
    sensitivityContentDetectionPolitic = SensitivityContentDetection(politic_model_path, sent_tokenizer)

    # 8 porn model
    porn_model_path = 'roberta_wwm_insult_model.onnx'
    sensitivityContentDetectionPorn = SensitivityContentDetection(porn_model_path, sent_tokenizer)

    # 9 violence model
    violence_model_path = 'roberta_wwm_insult_model.onnx'
    sensitivityContentDetectionViolence = SensitivityContentDetection(violence_model_path, sent_tokenizer)

    contentAudit = ContentAudit(ruleFilter, adDetection, deDuplication, sensitivityWordDetection,
                 sensitivityContentDetectionInsult, sensitivityContentDetectionPolitic, sensitivityContentDetectionPorn, sensitivityContentDetectionViolence)
    flag, sent = contentAudit.det('黑人很多都好吃懒做，偷奸耍滑！')





