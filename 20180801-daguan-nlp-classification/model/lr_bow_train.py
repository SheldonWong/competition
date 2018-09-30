
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

print('读取数据')
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

print('通过词袋模型表示文本特征：')
vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer.fit(df_train['word_seg'])

# 训练集
x_train = vectorizer.transform(df_train['word_seg'])
y_train = df_train['class'] - 1

# 测试集
x_test = vectorizer.transform(df_test['word_seg'])

print('开始训练')
lg = LogisticRegression(C=4,dual=True)
lg.fit(x_train,y_train)

#保存模型
import pickle 

with open('lg.pickle', 'wb') as f:
    pickle.dump(lg, f)


y_test = lg.predict(x_test)

df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id','class']]
df_result.to_csv('./result.csv', index=False)




