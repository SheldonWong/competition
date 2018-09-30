import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer

print('读取数据')
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

print('特征TF-IDF：')
#vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, 
  use_idf=True,smooth_idf=True, sublinear_tf=True,norm='l1')
vectorizer.fit(df_train['word_seg'])



# 训练的时候只用到词
x_train = vectorizer.transform(df_train['word_seg'])
#y_train = df_train['class'] - 1

x_test = vectorizer.transform(df_test['word_seg'])



# 序列化
print('序列化特征:')
#保存特征
import pickle 

with open('./tfidf/x_train3.pickle', 'wb') as f:
    pickle.dump(x_train, f)


with open('./tfidf/x_test3.pickle', 'wb') as f3:
    pickle.dump(x_test, f3)