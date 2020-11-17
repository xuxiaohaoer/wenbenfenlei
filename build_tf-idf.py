"""
1. 卡方检验 降维
2. TF-IDF
"""
from gensim import corpora
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
import pickle as pkl


len_dictionary = 10000  # dimension before chi2
len_features = 5000     # dimension after chi2  降维

x_train = corpora.MmCorpus('./bow/train_bow.mm')
y_train = np.load('./bow/y_train.npy')
x_test = corpora.MmCorpus('./bow/test_bow.mm')
y_test = np.load('./bow/y_test.npy')


# ========== 将词袋从gensim格式转换成scipy.sparse.csr_matrix格式
def convert(corpus):
    data = []
    rows = []
    cols = []
    row = 0
    for doc in corpus:
        for col, value in doc:
            rows.append(row)
            cols.append(col)
            data.append(value)
        row += 1
    matrix = csr_matrix((data, (rows, cols)))
    return matrix


x_train = convert(x_train)
x_test = convert(x_test)


# ====================== Chi-square test ==================
ch2 = SelectKBest(chi2, k=len_features)
x_train = ch2.fit_transform(x_train, y_train)
x_test = ch2.transform(x_test)
scores = ch2.scores_
# 保存降维结果
with open('./score/ch2_scores(to{}).pkl'.format(len_features), 'wb') as f:
    pkl.dump(scores, f)


# =========================== TF-IDF =====================
# TF-IDF计算权重
transformer = TfidfTransformer(smooth_idf=False)
x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

# 保存TF-IDF后的结果
with open('./tfidf/train_tfidf_{}.pkl'.format(len_features), 'wb') as f:
    pkl.dump(x_train, f)
with open('./tfidf/test_tfidf_{}.pkl'.format(len_features), 'wb') as f:
    pkl.dump(x_test, f)