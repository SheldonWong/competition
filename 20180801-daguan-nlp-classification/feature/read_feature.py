import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# =================================读入数据
# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv
print('读取数据')
df_train = pd.read_csv('~/workspace/sublime/daguan/train_sample.csv')
df_test = pd.read_csv('~/workspace/sublime/daguan/train_sample.csv')

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

# ==================================提取tfidf文本特征
# vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer.fit(df_train['word_seg'])

# 训练集
x_train = vectorizer.transform(df_train['word_seg'])
df_train['class'] = df_train['class'] - 1
y_train = df_train['class']

# 测试集
x_test = vectorizer.transform(df_test['word_seg'])

# 抽样，只用抽train的样本
df_train['feature'] = x_train
new_df = df_train[['feature','class']]
sub_sample = new_df.sample(n=5, frac=None, weights=None,replace=False, random_state=None, axis=0)
