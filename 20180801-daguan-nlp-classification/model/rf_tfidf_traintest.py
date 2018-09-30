from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import time
import pandas as pd


# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv
print('读取数据')
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/train_set.csv')

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

# 提取tfidf文本特征
# vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer = TfidfVectorizer()
vectorizer.fit(df_train['word_seg'])

# 训练集
x_train = vectorizer.transform(df_train['word_seg'])
y_train = df_train['class']

# 测试集
x_test = vectorizer.transform(df_test['word_seg'])


X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)


from sklearn.ensemble import RandomForestClassifier

start = time.time()
rf_clf = RandomForestClassifier(n_estimators=1000,max_depth=100,oob_score=True, random_state=666, n_jobs=-1)
rf_clf.fit(X_train, y_train)

print(rf_clf.oob_score_)

y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(classification_report(y_test, y_pred))
end = time.time()
print('time',end-start)