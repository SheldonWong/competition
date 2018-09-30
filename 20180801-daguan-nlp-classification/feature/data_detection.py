import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer

print('读取数据')
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')

# 文章长度分布

print()