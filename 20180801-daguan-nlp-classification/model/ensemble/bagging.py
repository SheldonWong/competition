import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm

'''
# =================================读入数据
# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv
print('读取数据')
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/train_set.csv')

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

print('特征TF-IDF：')
#vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, use_idf=1,smooth_idf=1, sublinear_tf=1)
vectorizer.fit(df_train['word_seg'])

# 训练的时候只用到词
x_train = vectorizer.transform(df_train['word_seg'])
y_train = df_train['class'] - 1

x_test = vectorizer.transform(df_test['word_seg'])
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

'''
train_X,test_X, train_y, test_y = train_test_split(x_train,
                                                   y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)
'''


print('开始用bagging训练:')

# 可选模型，LR，DecisionTree，SVM等
bagging_clf = BaggingClassifier(svm.LinearSVC(C=5,verbose=1),
                           n_estimators=5, max_samples=0.7,bootstrap=True,oob_score=True)
bagging_clf.fit(x_train, y_train)



# 在训练集上的误差
a_score = bagging_clf.score(train_X, train_y)
print('训练集上的误差:{}'.format(a_score))

# 在袋外数据的误差
oob_error = bagging_clf.oob_score_
print('oob_score_:{}'.format(oob_error))

# 在测试集上的预测结果与保存
y_test = bagging_clf.predict(x_test)

df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id','class']]
#df_result.to_csv('./result/lr-ensemble.csv', index=False)