#encoding=utf8
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

 
'''
# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv
print('read')
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_train['class'] = df_train['class'] - 1
df_train[df_train['class'] > 0] = 1
'''





'''
df_test = pd.read_csv('~/Downloads/train_set.csv')

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

print('特征TF-IDF：')
#vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, use_idf=True,smooth_idf=True, sublinear_tf=True)
vectorizer.fit(df_train['word_seg'])



# 训练的时候只用到词
x_train = vectorizer.transform(df_train['word_seg'])
y_train = df_train['class']

x_test = vectorizer.transform(df_test['word_seg'])
'''




'''
# 序列化
print('序列化特征:')
#保存特征
import pickle 

with open('./feature/x_train.pickle', 'wb') as f:
    pickle.dump(x_train, f)

with open('./feature/y_train.pickle', 'wb') as f2:
    pickle.dump(y_train, f2)

with open('./feature/x_test.pickle', 'wb') as f3:
    pickle.dump(x_test, f3)

print('读取特征:')
#读取Model
import pickle

with open('./feature/x_train.pickle', 'rb') as f:
    x_train = pickle.load(f)

df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')
y_train = df_train['class'] - 1
#y_train.to_csv('./feature/y_train.csv')

with open('./feature/x_test.pickle', 'rb') as f3:
    x_test = pickle.load(f3)

'''



import pickle

print('load features:')
with open('./feature/tfidf/x_train2.pickle', 'rb') as f:
    x_train = pickle.load(f)

df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')
df_train['class'] = df_train['class'] - 1
df_train[df_train['class'] > 0] = 1
y_train = df_train['class']
#y_train.to_csv('./feature/y_train.csv')

with open('./feature/tfidf/x_test2.pickle', 'rb') as f3:
    x_test = pickle.load(f3)

train_X,test_X, train_y, test_y = train_test_split(x_train,
                                                   y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)
print('start fit')
lg = LogisticRegression(C=5,dual=True,verbose=1)
lg.fit(train_X,train_y)


'''
#保存模型
import pickle 

with open('./model/lg2.pickle', 'wb') as f:
    pickle.dump(lg, f)


#读取Model

with open('./model/lg2.pickle', 'rb') as f:
    lg = pickle.load(f)
'''


y_pred= lg.predict(test_X)
accuracy = accuracy_score(test_y, y_pred)
print(accuracy)
print(classification_report(test_y, y_pred))


y_test = lg.predict(x_test)

df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id','class']]
#df_result.to_csv('./result/result-tfidf-multinomial.csv', index=False)
