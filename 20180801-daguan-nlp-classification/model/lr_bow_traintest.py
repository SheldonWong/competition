import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


print('读取数据')
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

print('特征：')
vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer.fit(df_train['word_seg'])

# 训练的时候只用到词
X = vectorizer.transform(df_train['word_seg'])
# x_test = vectorizer.transform(df_test['word_seg'])
y = df_train['class'] - 1

train_X,test_X, train_y, test_y = train_test_split(X,
                                                   y,
                                                   test_size = 0.2,
                                                   random_state = 0)
lg = LogisticRegression(C=4,dual=True)
lg.fit(train_X,train_y)

'''
#保存模型
import pickle 

with open('lg2.pickle', 'wb') as f:
    pickle.dump(lg, f)


#读取Model
import pickle 
with open('lg2.pickle', 'rb') as f:
    lg = pickle.load(f)
'''

y_pred= lg.predict(test_X)
accuracy = accuracy_score(test_y, y_pred)
print(accuracy)
print(classification_report(test_y, y_pred))