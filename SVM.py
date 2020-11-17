"""
训练并保存模型用
"""
import pickle as pkl
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics


# ============= training svm models =============
def linersvm(x, y, model_dir='./model/linersvm.pkl'):
    clf = LinearSVC()
    clf.fit(x, y)

    with open(model_dir, 'wb') as f:
        pkl.dump(clf, f)

    print('the metrics of linersvm on train set: ')


if __name__ == '__main__':
    len_features = 5000

    with open('./tfidf/train_tfidf_{}.pkl'.format(len_features), 'rb') as f:
        tfidf = pkl.load(f)
    y = np.load('./data/y_train.npy')
    assert tfidf.shape[0] == y.shape[0]
    linersvm(tfidf, y)
