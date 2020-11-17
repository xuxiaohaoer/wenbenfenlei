import os
from gensim.sklearn_api import TfIdfTransformer
from gensim import corpora
import dic
import numpy as np

src_dir = "./bow/"
"""
这里可能会产生问题 fit的参数问题？
"""
# 读入词袋
train_bow = corpora.MmCorpus(src_dir + "train_bow.mm")
test_bow = corpora.MmCorpus(src_dir + "test_bow.mm")
data_bow = corpora.MmCorpus(src_dir + "dataset_bow.mm")
# 读入词典
dictionary = dic.build_dic()
print("加载完成")
model = TfIdfTransformer(dictionary=dictionary).fit(data_bow)
print("正在转换")
test = train_bow[0:100]
print(train_bow)
train_tfidf = model.transform(test)
print((test))
print((train_tfidf))
test_tfidf = model.transform(test_bow)
print(type(test_tfidf))
