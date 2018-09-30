
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm, datasets
from sklearn.calibration import CalibratedClassifierCV

import pickle

print('载入特征:')
with open('./feature/tfidf/x_train3.pickle', 'rb') as f:
    x_train = pickle.load(f)

df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')
y_train = df_train['class'] - 1
#y_train.to_csv('./feature/y_train.csv')

with open('./feature/tfidf/x_test3.pickle', 'rb') as f3:
    x_test = pickle.load(f3)

word_len = pd.read_csv('./feature/tfidf/word_len.csv')
test_len = pd.read_csv('./feature/tfidf/test_word_len.csv') 


from scipy.sparse import coo_matrix,csr_matrix,hstack

# 把一个列向量拼接到csr矩阵的最后一列
# a 稀疏矩阵
# b 列向量
def concat(a,b):
	row = np.array(range(a.shape[0]))
	col = np.array([0]*a.shape[0])
	data = b['word_len'].values
	b = csr_matrix((data, (row, col)), shape=(a.shape[0], 1))

	res = hstack((a,b))
	return res.tocsr()

x_train = concat(x_train,word_len)

train_X,test_X, train_y, test_y = train_test_split(x_train,
                                                   y_train,
                                                   test_size = 0.1,
                                                   random_state = 0)
print('开始用SVM训练')
### 仅针对RBF核函数，gamma大，方差小，方差很小的高斯分布长得又高又瘦，训练准确率高
### C针对所有核，C越大，kesi越小，越不能容忍错分，当C无穷大时，线性SVM又变成线性可分SVM，
### C越小，kesi越大，越能容忍错分的样本

import time


start = time.time()
#用全部数据训练
lin_svc = svm.LinearSVC(C=5).fit(train_X, train_y) #线性核
'''
lin_svc = svm.SVC(kernel='rbf', gamma=0.7, C=0.0).fit(train_X, train_y) # 径向基核
'''
end = time.time()
print(end-start)


# 在测试集上的report
y_pred= lin_svc.predict(test_X)
accuracy = accuracy_score(test_y, y_pred)
print(accuracy)
print(classification_report(test_y, y_pred))


# 结果预测与保存
x_test = concat(x_test,test_word_len)
y_test = lin_svc.predict(x_test)

df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id','class']]
df_result.to_csv('./result/linsvm-tfidflen-traintest.csv', index=False)