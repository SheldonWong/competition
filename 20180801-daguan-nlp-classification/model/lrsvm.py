import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

'''
# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv
print('读取数据')
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/train_set.csv')

df_train.drop(columns=['word_seg','id'], inplace=True)
df_test.drop(columns=['word_seg'], inplace=True)

print('特征TF-IDF：')
#vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9, 
  use_idf=True,smooth_idf=True, sublinear_tf=True,norm='l2')
vectorizer.fit(df_train['article'])



# 训练的时候只用到词
x_train = vectorizer.transform(df_train['article'])
y_train = df_train['class'] - 1

x_test = vectorizer.transform(df_test['article'])

'''

import pickle

print('载入特征:')
with open('./feature/tfidf/x_train2.pickle', 'rb') as f:
    x_train = pickle.load(f)

df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')
y_train = df_train['class'] - 1
#y_train.to_csv('./feature/y_train.csv')

with open('./feature/tfidf/x_test2.pickle', 'rb') as f3:
    x_test = pickle.load(f3)


train_X,test_X, train_y, test_y = train_test_split(x_train,
                                                   y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)

#test_X是稀疏矩阵

print('开始SVM训练')
lin_svc = CalibratedClassifierCV(base_estimator=svm.LinearSVC(C=5)).fit(x_train, y_train)
lin_svc.fit(x_train,y_train)

print('开始用LR训练')
# C越大，惩罚越小
lg = LogisticRegression(C=5,dual=True,verbose=1)
lg.fit(x_train,y_train)



y_prob = lg.predict_proba(x_test)
y_prob2 = lin_svc.predict_proba(x_test)

result1 = pd.read_csv('result1.csv')
y_prob1 = result1['prob']

result2 = pd.read_csv('result2.csv')
y_prob2 = result2['prob']

y_prob3 = alpha1 * y_prob1 + alpha2 * y_prob2


y_class = [np.argmax(row) for row in y_prob3]


df_test['class'] = y_class
df_test['class'] = df_test['class'] + 1
df_test['prob'] = y_prob3.tolist()
df_result = df_test.loc[:, ['id','class','prob']]
df_result.to_csv('./result/lrsvm.csv', index=False)


