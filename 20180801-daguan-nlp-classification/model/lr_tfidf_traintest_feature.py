import pandas as pd 
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


print('开始用LR训练')
# C越大，惩罚越小
lg = LogisticRegression(C=5,dual=True,verbose=1)
lg.fit(train_X,train_y)


y_pred= lg.predict(test_X)
accuracy = accuracy_score(test_y, y_pred)

y_prob = lg.predict_proba(test_X)

print(accuracy)
print(classification_report(test_y, y_pred))


print('开始SVM训练')
svc = svm.LinearSVC(C=5,dual=True)
lin_svc = CalibratedClassifierCV(base_estimator=svc)
lin_svc.fit(x_train,y_train)

y_pred2= lin_svc.predict(test_X)
accuracy = accuracy_score(test_y, y_pred2)

y_prob2 = lin_svc.predict_proba(test_X)

print(accuracy)
print(classification_report(test_y, y_pred2))
# bad case 

# 特征
for row in test_X:
    freature_l.append(str(row).replace('\t',' ').replace('\n',' '))

case_df = pd.DataFrame(columns=['feature','class','pred','pred2','prob','prob2'])

case_df['feature'] = freature_l
case_df['class'] = test_y.tolist()
case_df['pred'] = y_pred
case_df['pred2'] = y_pred2
case_df['prob'] = y_prob.tolist()
case_df['prob2'] = y_prob2.tolist()

case_df.to_csv('./result/bad_case3.csv')

## 获取badcase索引
b = case_df['class'] == case_df['pred']
b_l = list(b)
index_l = [i for i in range(len(b_l)) if b_l[i] == False]


c = case_df['class'] == case_df['pred2']
c_l = list(c)
index2_l = [i for i in range(len(c_l)) if c_l[i] == False]


# 二者预测结果不同的列表
d = case_df['pred'] == case_df['pred2']
d_l = list(d)
index3_l = [i for i in range(len(d_l)) if d_l[i] == False]

ana_df = pd.DataFrame(columns=['t-l','t-s','l-s'])
ana_df['t-l'] = index_l
ana_df['t-s'] = index2_l
ana_df['l-s'] = index3_l
ana_df.to_csv('./result/bad_case3_ana.csv')

'''
y_class = lg.predict(x_test)
y_prob = lg.predict_proba(x_test)




df_test['class'] = y_class.tolist()
df_test['class'] = df_test['class'] + 1
df_test['prob'] = y_prob.tolist()
df_result = df_test.loc[:, ['id','class','prob']]
df_result.to_csv('./result/result-tfidf-feature-prob.csv', index=False)
'''