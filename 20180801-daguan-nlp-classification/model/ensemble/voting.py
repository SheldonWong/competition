import pandas as pd 
import numpy as np 

'''
lin_svm_df = pd.read_csv('lin_svm-tfidf-C01.csv')
lin_svm_df2 = pd.read_csv('lin_svm-tfidf-C10.csv')
lr_tfidf = pd.read_csv('lr-tfidf-train.csv')
fasttext_tfidf = pd.read_csv('result-fasttext.csv')
result_tfidf_C5 = pd.read_csv('result-tfidf-C5.csv')
result_tfidf_feature = pd.read_csv('result_tfidf_feature')
result_tfidf_multinomial = pd.read_csv('result-tfidf-multinomial.csv')
svm_tfidf = pd.read_csv('svm-tfidf.csv')

lin_svm_df['class1'] = lin_svm_df2['']
'''

df_voting = pd.read_csv('result/voting.csv')

#weight = [2.5,2,2]

res = []
# 投票，如果
for row in df_voting.values:
	if(row[1] == row[2]):
		res.append(row[1])
	else:
		res.append(row[0])

df_test = pd.read_csv('~/Downloads/train_set.csv')
df_test.drop(columns=['word_seg'], inplace=True)
df_test['class']= res
df_result = df_test.loc[:, ['id','class']]
df_result.to_csv('./result/voting.csv', index=False)