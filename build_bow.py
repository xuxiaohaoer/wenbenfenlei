from gensim import corpora
import os
import load
import dic
import numpy as np

dst_str = "./bow/"


def build_bow(dictory, dataset, datasetstr):
    if os.path.exists(dst_str + datasetstr + "_bow.mm"):
        # 读取生成好的词袋
        print("读取" + datasetstr + "中")
        bow = corpora.MmCorpus(dst_str + datasetstr + "_bow.mm")
    else:
        # 生成词袋
        bow = [dictory.doc2bow(doc) for doc in dataset]
        corpora.MmCorpus.serialize(dst_str + datasetstr + "_bow.mm", corpus=bow)
    return bow


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, dataset, test = load.loda_file("./data/")
    # 读取/生成字典
    dictory = dic.build_dic()

    # 读取/生成词袋
    train_bow = build_bow(dictory, x_train, datasetstr="train")
    test_bow = build_bow(dictory, x_test, datasetstr="test")
    dataset = build_bow(dictory, dataset, datasetstr="dataset")
    test = build_bow(dictory, test, datasetstr="test")

    print(type(train_bow), len(train_bow))

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    np.save(dst_str + "y_train.npy", y_train)
    np.save(dst_str + "y_test.npy", y_test)
