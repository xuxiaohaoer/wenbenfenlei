"""
模型保存好后，展示效果用
"""
import pickle as pkl
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics

len_features = 5000
model_dir = "./model/svm.pkl"

print("read model")
with open(model_dir, 'rb') as f:
    clf = pkl.load(f)

print("read test dataset")
with open('./tfidf/test_tfidf_{}.pkl'.format(len_features), 'rb') as f:
    x = pkl.load(f)

y = np.load('./data/y_test.npy')

print("begin predict")
y_pred = clf.predict(x)

# 输出结果
print(metrics.accuracy_score(y, y_pred))
print(metrics.confusion_matrix(y, y_pred))
