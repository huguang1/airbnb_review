from numpy import mean, median

import random
import pandas as pd
import numpy as np
import time
from keras.layers import Dense, Flatten, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras import layers

review = pd.read_csv('review_translate_all_delete_low_6751.csv', encoding='utf-8')
listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
df1 = listings.drop(listings[np.isnan(listings['review_scores_rating'])].index)
df2 = df1.drop(df1[np.isnan(df1['review_scores_accuracy'])].index)
df3 = df2.drop(df2[np.isnan(df2['review_scores_cleanliness'])].index)
df4 = df3.drop(df3[np.isnan(df3['review_scores_checkin'])].index)
df5 = df4.drop(df4[np.isnan(df4['review_scores_communication'])].index)  # 最高
df6 = df5.drop(df5[np.isnan(df5['review_scores_location'])].index)
listings = df6.drop(df6[np.isnan(df6['review_scores_value'])].index)
a = time.time()
list_review = []
list_score1 = []
dict_score = {}
# 统计每个房子的得分
for _, row in listings.iterrows():
    dict_score[int(row['id'])] = row['review_scores_rating']
# 对每个房子和评论进行匹配
for _, row in review.iterrows():
    if type(row['comments']) is not float and row['listing_id'] in dict_score.keys():
        list_score1.append(dict_score[row['listing_id']])
        list_review.append(row['comments'])

list_score = []
a_mean = mean(list_score1)
for i in list_score1:
    if i >= a_mean:
        list_score.append(1)
    else:
        list_score.append(0)

# print(aa)
# print(bb)
# print(cc)
# define documents
docs = list_review

# define class labels
labels = np.array(list_score)
# integer encode the documents
vocab_size = 5000
encoded_docs = [one_hot(d, vocab_size) for d in docs]  # one_hot编码到[1,n],不包括0

# pad documents to a max length of 4 words
max_length = 2000
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


# define the model
input = Input(shape=(2000,))
x = Embedding(vocab_size, 8, input_length=max_length)(input)  # 这一步对应的参数量为50*8
# x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
# x = layers.Bidirectional(layers.LSTM(8))(x)
# x = layers.Dense(64, activation='relu')(x)
# model.add(layers.Dense(1, activation='sigmoid'))
# x = layers.Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input, outputs=x)
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
Xtrain, Xtest, ytrain, ytest = train_test_split(padded_docs, labels, test_size=0.2)
# fit the model
# model.fit(padded_docs, labels, epochs=100, verbose=0)
history = model.fit(Xtrain, ytrain, epochs=3, batch_size=32, validation_data=(Xtest, ytest))
y_predict = model.predict(Xtest)
# evaluate the model
# loss, accuracy = model.evaluate(Xtest, ytest, verbose=0)
# loss_test, accuracy_test = model.evaluate(Xtest, ytest, verbose=0)
# print('Accuracy: %f' % (accuracy * 100))

# dummy = DummyClassifier(strategy="most_frequent").fit(padded_docs, labels)
# ydummy = dummy.predict(padded_docs)
# print(confusion_matrix(labels, ydummy))
# print(classification_report(labels, ydummy))
a = []
for i in y_predict:
    print(type(i[0]))
y_predict = [round(i[0]) for i in y_predict]
print(confusion_matrix(ytest, y_predict))
print(classification_report(ytest, y_predict))


plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
# Xtrain, Xtest, ytrain, ytest = train_test_split(features, z, test_size=0.2)

dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
fpr, tpr, _ = roc_curve(ytest, dummy.predict_proba(Xtest)[:, 1])
plt.plot(fpr, tpr, c='b', label='baseline most_frequent', linestyle='-.')

model.fit(Xtrain, ytrain, epochs=3, batch_size=32, validation_data=(Xtest, ytest))
predict_z = model.predict(Xtest)
fpr, tpr, _ = roc_curve(ytest, predict_z)
plt.plot(fpr, tpr, c='g', label='embedding with MLP')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')
plt.show()


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

# epochs = range(1, len(loss_values) + 1)
# plt.plot(epochs, loss_values, 'ro', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
#
# plt.plot(epochs, acc_values, 'ro', label='Training accuracy')
# plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


