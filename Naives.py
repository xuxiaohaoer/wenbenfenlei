"""
朴素贝叶斯训练过程
"""
import pickle as pkl
import numpy as np
from scipy import sparse


len_features = 5000
if __name__ == "__main__":
    with open("./tfidf/train_tfidf_{}.pkl".format(len_features), 'rb') as f:
        tfidf = pkl.load(f)
    y = np.load('./bow/y_train.npy')
    # ============== 计算条件概率 ==============
    print("begin calculate")
    scores = np.ones((len(np.unique(y)), tfidf.shape[1]))
    for i in range(len(y)):
        doc = tfidf[i].toarray().flatten()
        scores[y[i]] += doc
    rowsum = scores.sum(1)
    r_inv = np.power(rowsum, -1).flatten()
    r_mat_inv = sparse.diags(r_inv)
    scores = np.log(r_mat_inv.dot(scores))
    print("calculate end")
    # ============== 保存模型 ===============
    print('saving NBscores_{}.npy'.format(len_features))
    np.save('./data/NBscores_{}.npy'.format(len_features), scores)