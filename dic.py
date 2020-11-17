# coding=utf-8
import os
from gensim import corpora

from getext import get_texts, get_alltexts

base_dir = "./data/"
dst_dir = "./dicdata/"
len_dic = 100000


def build_dic():
    if (os.path.exists(dst_dir + "dictionary" + str(len_dic))):
        # 读取字典
        print("读取字典中")
        dic = corpora.Dictionary.load((dst_dir + "dictionary" + str(len_dic)))
    else:
        # 创建字典
        print("创建字典中")
        texts = get_alltexts(base_dir)
        dic = corpora.Dictionary(texts)
        dic.filter_extremes(keep_n=100000)
        dic.save(dst_dir + "dictionary" + str(len_dic))
        dic.save_as_text(dst_dir + "dictionary" + str(len_dic) + ".txt")
    return dic
