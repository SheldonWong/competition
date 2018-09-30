# 拼接特征
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer

# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv
train_path = '~/workspace/sublime/daguan/train_sample.csv'
test_path = '~/workspace/sublime/daguan/train_sample.csv'

word_len = pd.read_csv('./tfidf/word_len.csv')

word_len = word_len[:10]
print(word_len)

print('read data')
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

print('TF-IDF')
#vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, 
  use_idf=True,smooth_idf=True, sublinear_tf=True)

vectorizer.fit(df_train['word_seg'])
#vectorizer.fit(np.hstack((df_train['word_seg'].data,df_test['word_seg'].data)))


# 训练的时候只用到词
x_train = vectorizer.transform(df_train['word_seg'])

#y_train = df_train['class'] - 1
#x_test = vectorizer.transform(df_test['word_seg'])

print(x_train.shape)
# 拼接
from scipy.sparse import coo_matrix,csr_matrix,hstack
import numpy as np 

print(type(x_train))
#row = np.array([0,1,2,3,4,5,6,7,8,9])
# 稀疏矩阵拼接方案
# 把一个列向量b拼接到a的后面，前提是二者的行数相等，如果不等，补零？
#def concat(a,b):
row = np.array(range(x_train.shape[0]))
col = np.array([0]*x_train.shape[0])
data = word_len['word_len'].values
b = csr_matrix((data, (row, col)), shape=(10, 1))

print(b)
res = hstack((x_train,b)).toarray()
print(res)
print(res.shape)

# 将列向量文章长度len，拼接到稀疏矩阵的最后一列
# a = x_train,默认输出的tfidf向量，csr格式
# b = word_len, word_len = pd.read_csv('word_len.csv')

def concat(a,b):
	row = np.array(range(a.shape[0]))
	col = np.array([0]*a.shape[0])
	data = b['word_len'].values
	b = csr_matrix((data, (row, col)), shape=(a.shape[0], 1))

	res = hstack((a,b))
	return res.tocsr()
	
r = concat(x_train,word_len)
print(type(r))

