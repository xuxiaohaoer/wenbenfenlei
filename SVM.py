"""
训练并保存模型用
"""
import pickle as pkl
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics


# ============= training svm models =============
def linersvm(x, y, model_dir='./model/linersvm.pkl'):
    print("model begin")
    clf = LinearSVC()
    clf.fit(x, y)

    with open(model_dir, 'wb') as f:
        pkl.dump(clf, f)

    print('model end')


if __name__ == '__main__':
    len_features = 5000
    # ============== 读入数据 ===============
    print("data read begin")
    with open('./tfidf/train_tfidf_{}.pkl'.format(len_features), 'rb') as f:
        tfidf = pkl.load(f)
    y = np.load('./bow/y_train.npy')
    print("data read end")
    assert tfidf.shape[0] == y.shape[0]
    linersvm(tfidf, y)
