import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


### 载入特征
import pickle

print('载入特征:')
with open('./feature/x_train.pickle', 'rb') as f:
    x_train = pickle.load(f)

df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')
y_train = df_train['class'] - 1
#y_train.to_csv('./feature/y_train.csv')

with open('./feature/x_test.pickle', 'rb') as f3:
    x_test = pickle.load(f3)




### LR训练
print('开始用LR训练')
lg = LogisticRegression(C=5,dual=True,
						multi_class='ovr',
						verbose=1,class_weight=None)
lg.fit(x_train,y_train)

'''
#保存模型
import pickle 

with open('./model/lg2.pickle', 'wb') as f:
    pickle.dump(lg, f)


#读取Model

with open('./model/lg2.pickle', 'rb') as f:
    lg = pickle.load(f)

'''

### 在训练集上的表现
y_pred= lg.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
print(accuracy)
print(classification_report(y_train, y_pred))


y_test = lg.predict(x_test)

df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id','class']]
df_result.to_csv('./result/lr-tfidf-train.csv', index=False)