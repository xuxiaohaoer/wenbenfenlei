"""
朴素贝叶斯预测过程
"""
import pickle as pkl
import numpy as np
from scipy import sparse
from sklearn import metrics

len_features = 5000
# ========== 开始读入数据 ===========
print("data read begin")
with open("./tfidf/test_tfidf_{}.pkl".format(len_features), 'rb') as f:
    tfidf = pkl.load(f)
y = np.load('./bow/y_test.npy')
print("data read end")
# ========== 结束读入数据 ===========

scores = np.load('./data/NBscores_{}.npy'.format(len_features))
y_pred = []
i = 1
for doc in tfidf:
    doc = doc.toarray().flatten()
    # for i, key in enumerate(doc):
    #     if key != 0:
    #         doc[i] = 1
    tmp = scores.dot(doc)
    y_pred.append(tmp.argmax())
print(metrics.accuracy_score(y, y_pred))
print(metrics.confusion_matrix(y, y_pred))
print(metrics.f1_score(y, y_pred, average="macro"))