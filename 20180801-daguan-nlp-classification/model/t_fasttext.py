import fasttext
import pandas as pd 


'''
# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv

### read data
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

### transfer data to match input
df_train = df_train[['class','word_seg']]
#df_train['class'] = '__label__' + df_train['class']

with open("data/train.txt", "w") as output:
    for row in df_train.values:
        output.write('__label__'+str(row[0]) + ' ' + row[1] + '\n')

with open("data/test.txt", "w") as output:
    for row in df_test['word_seg'].values:
        output.write(row + '\n')

'''

'''
### split train.txt to trainset.txt and testset.txt

import random

with open('./data/train.txt', 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)

l = int(len(lines)*0.8)
#80%作为训练集
with open('./data/trainset.txt', 'w') as fa:
    for row in lines[:l]:
        fa.write(row)
#20%作为测试集
with open('./data/testset.txt', 'w') as fb:
    for row in lines[l:]:
        fb.write(row)   

'''

# fit and predict
df_test = pd.read_csv('~/Downloads/test_set.csv')
df_test.drop(columns=['article'], inplace=True)

classifier=fasttext.supervised('./data/train.txt','./model/daguan-classifier',
                                lr=0.5,lr_update_rate=10,dim=120,ws=5,epoch=10,
                                min_count=10,word_ngrams=1,t=0.0001)
result = classifier.test('./data/testset.txt')

print(result.precision)
print(result.recall)
f1 = 2/(1/result.precision+1/result.recall)
print(f1)


with open('./data/test.txt','r') as f:
    lines = f.readlines()
    result = classifier.predict(lines)
    proba = classifier.predict_proba(lines,k=19)



pred_l = [e[0] for e in result]
proba_l = proba
df_test['class'] = pred_l
df_test['proba'] = proba_l

df_result = df_test.loc[:, ['id','class']]
df_result.to_csv('./result/fasttext—word2vec-result-0.762.csv', index=False)



df_result2 = df_test.loc[:, ['id','proba']]
df_result2.to_csv('./result/fasttext—word2vec-proba-0.762.csv', index=False)


'''参数
input_file                 训练文件路径（必须）
output                     输出文件路径（必须）
label_prefix               标签前缀 default __label__
lr                         学习率 default 0.1
lr_update_rate             学习率更新速率 default 100
dim                        词向量维度 default 100
ws                         上下文窗口大小 default 5
epoch                      epochs 数量 default 5
min_count                  最低词频 default 5
word_ngrams                n-gram 设置 default 1
loss                       损失函数 {ns,hs,softmax} default softmax
minn                       最小字符长度 default 0
maxn                       最大字符长度 default 0
thread                     线程数量 default 12
t                          采样阈值 default 0.0001
silent                     禁用 c++ 扩展日志输出 default 1
encoding                   指定 input_file 编码 default utf-8
pretrained_vectors         指定使用已有的词向量 .vec 文件 default None
'''
