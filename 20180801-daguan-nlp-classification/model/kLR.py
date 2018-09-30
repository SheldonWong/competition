from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np 
import pandas as pd 

batch_size = 4000
num_classes = 19
epochs = 10

# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv
print('read data')
df_train = pd.read_csv('~/workspace/sublime/daguan/train_sample.csv')
df_test = pd.read_csv('~/workspace/sublime/daguan/train_sample.csv')

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

print('feature:')
#vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, 
  use_idf=True,smooth_idf=True, sublinear_tf=True,max_features=100000)
vectorizer.fit(df_train['word_seg'])



# 训练的时候只用到词
x_train = vectorizer.transform(df_train['article'])
y_train = df_train['class'] - 1
x_test = vectorizer.transform(df_test['article'])




train_X,test_X, train_y, test_y = train_test_split(x_train,
                                                   y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)


train_y = keras.utils.to_categorical(train_y, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)

print(train_X.shape)


# 模型 x_train.shape[1]
model = Sequential()
model.add(Dense(10000, activation='relu', input_shape=(train_X.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(train_X, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(test_X, test_y))

model.save('mlp.h5')

# 评价
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
