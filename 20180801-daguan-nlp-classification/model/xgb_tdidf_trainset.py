import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



'''
# =================================读入数据
# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv
print('读取数据')
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/train_set.csv')

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
sub_sample = new_df.sample(n=5, frac=None, weights=None,replace=True, random_state=None, axis=0)

'''

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


#
train_X,test_X, train_y, test_y = train_test_split(x_train,
                                                   y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)
# print(train_X[0])

# ===================================模型配置
import xgboost as xgb
dtrain=xgb.DMatrix(train_X,label=train_y)
# for test
d_test = xgb.DMatrix(test_X)

# for save
dtest=xgb.DMatrix(x_test)

params={'booster':'gbtree',
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'max_depth':100,
    'lambda':10,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':2,
    'eta': 0.05,
    'seed':0,
    'nthread':8,
     'silent':1,
 	'num_class':19}

watchlist = [(dtrain,'train')]

# ====================================训练
print('xgb训练')
bst = xgb.train(params,dtrain,num_boost_round=200,evals=watchlist)

# ====================================保存模型


# ====================================report
y_pred_1 = bst.predict(d_test)

print('ACC: %.4f' % accuracy_score(test_y,y_pred_1))
print(classification_report(test_y,y_pred_1))


# ====================================预测并保存结果
y_pred = bst.predict(dtest)


# 保存
df_test['class'] = y_pred.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id','class']]
df_result.to_csv('./result/result-tfidf-xgb.csv', index=False)











