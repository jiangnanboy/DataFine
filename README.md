### DataFine 数据语料清洗

-----------------------------------------------------------------------
数据是人工智能领域发展的基础要素之一。随着大规模预训练模型及相关技术不断取得突破，在相应研究中使用高效数据处理工具提升数据质量变得越来越重要。
 
DataFine主要包含规则清洗、敏感词过滤、广告过滤、去重以及敏感内容等功能在内的多个数据处理方法，为中文语料的训练提供安全可靠的数据。

Data is one of the fundamental elements of the development of artificial intelligence.

With the continuous breakthrough of large-scale pre-training models and related technologies, 

it is becoming more and more important to use efficient data processing tools to improve data quality in corresponding research. 

DataFine mainly includes a number of data processing methods including rule cleaning, sensitive word filtering, advertisement filtering, 

de-duplication and sensitive content functions, providing safe and reliable data for the training of Chinese corpus.

本项目由python实现，java实现请移步到https://github.com/jiangnanboy/llm_corpus_quality。

DataFine支持以下方法：

* 规则清洗(rule cleaning)

* 敏感词过滤(sensitive word filtering)

* 广告过滤(advertising word filtering)

* 去重(deduplication)

* 敏感内容过滤(sensitive content filtering)
--------------------------------------------------------------------------------

语料清洗流程，目前共包括5个模块：

Corpus cleaning process currently consists of 5 modules:

1. 规则清洗：利用规则对一些低质量的文本段落进行初步过滤，这些规则主要包括低密度文本、异常符号、中文比例过低等。

   rule cleaning: Some low-quality text paragraphs are preliminatively filtered by rules, which mainly include low-density text, abnormal symbols, and low Chinese proportion.          

2. 敏感词过滤器：利用自动机，过滤色情、赌博、敏感等内容的文本。

   sensitive word filtering: Filter text for pornography, gambling, sensitive content, etc.                       

3. 广告过滤：利用textcnn模型，过滤涉嫌广告内容。(见https://github.com/jiangnanboy/ad_detect_textcnn)

   advertising word filtering:Filter advertising content.                         

4. 去重：利用simhash对相似文本片段进行去重。

   deduplication: simhash is used to de-duplicate similar text fragments.

5. 敏感内容过滤：采用roberta模型训练的文本分类方法，主要包括

(1).政治类(politics detection)

(2).暴恐类(violence detection)

(3).色情类(porn detection)

(4).辱骂歧视类(insult detection)

   Sensitive content filtering: Text classification methods trained by roberta model mainly include:
   
   (1).politics detection

   (2).violence detection

   (3).porn detection

   (4).insult detection

### usage
【main.py】

* 规则清洗(rule cleaning) -> src/rule
* 敏感词过滤(sensitive word filtering) -> src/sensitivity_word
* 广告过滤(advertising word filtering) -> src/advertising
* simhash去重(deduplication) -> src/deduplication
* 敏感内容过滤(sensitive content filtering) -> src/sensitivity_content

(model weights download: https://huggingface.co/jiangnanboy/content_audit)

``` python
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

```

### todo
利用LLM做内容质量评估。

### requirement
simhash

jieba

pickle

onnxruntime

transformers

### contact
1、github：https://github.com/jiangnanboy

2、blog：https://www.cnblogs.com/little-horse/

3、e-mail:2229029156@qq.com

### reference
https://github.com/hailin0/sensitive-word-filter

https://github.com/xlturing/Simhash4J

https://github.com/jiangnanboy/java_textcnn_onnx

https://github.com/jiangnanboy/llm_security

https://github.com/jiangnanboy/llm_corpus_quality
