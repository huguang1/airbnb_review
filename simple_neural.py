import random
import pandas as pd
from keras.datasets import imdb
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras import models
from keras import layers
import numpy as np
import time
from keras.layers import LSTM
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


review = pd.read_csv('review_translate_all_delete_low_432.csv', encoding='utf-8')
listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
df1 = listings.drop(listings[np.isnan(listings['review_scores_rating'])].index)
df2 = df1.drop(df1[np.isnan(df1['review_scores_accuracy'])].index)
df3 = df2.drop(df2[np.isnan(df2['review_scores_cleanliness'])].index)
df4 = df3.drop(df3[np.isnan(df3['review_scores_checkin'])].index)
df5 = df4.drop(df4[np.isnan(df4['review_scores_communication'])].index)
df6 = df5.drop(df5[np.isnan(df5['review_scores_location'])].index)
listings = df6.drop(df6[np.isnan(df6['review_scores_value'])].index)
a = time.time()
list_review = []
list_score = []
dict_score = {}
# 统计每个房子的得分
for _, row in listings.iterrows():
    dict_score[int(row['id'])] = row['review_scores_communication']
# 对每个房子和评论进行匹配
for _, row in review.iterrows():
    if type(row['comments']) is not float and row['listing_id'] in dict_score.keys():
        list_score.append(dict_score[row['listing_id']])
        list_review.append(row['comments'])
# 将评论写成向量
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(list_review)
# label_encoder = preprocessing.LabelEncoder()
# list_score = label_encoder.fit_transform(list_score)
# 使用逻辑回归进行预测
Xtrain, Xtest, ytrain, ytest = train_test_split(X.toarray(), list_score, test_size=0.2)
y_train = np.array(ytrain)
y_test = np.array(ytest)


# 模型由两个中间层（每层16个隐藏单元）和一个输出层（输出标量）组成
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(386,)))
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(units=1, activation='linear'))

# 编译模型，使用内置的优化器、损失函数
model.compile(optimizer='adam', loss='mse')

# 验证数据
x_val = Xtrain[:2000]
partial_x_train = Xtrain[2000:]
y_val = y_train[:2000]
partial_y_train = y_train[2000:]

# 训练模型，每批次512条样本，共20次迭代，把训练过程中的数据保存在history
history = model.fit(partial_x_train, partial_y_train, epochs=30, batch_size=32, validation_data=(x_val, y_val))

# y_predict = model.predict(Xtest)
# x = []
# for i in y_predict:
#     if i >= 5:
#         x.append(5)
#     elif i <= 0:
#         x.append(0)
#     else:
#         x.append(i)
# r_score = r2_score(y_true=ytest, y_pred=x)
# print(r_score)
# history_dict = history.history
# loss_values = history_dict['loss'][10:]
# val_loss_values = history_dict['val_loss'][10:]
#
# epochs = range(1, len(loss_values) + 1)
#
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()

# 绘制训练精度和验证精度
# plt.clf()  # clear the pic
# acc = history_dict['accuracy']
# val_acc = history_dict['val_accuracy']
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title("Training and validation accuracy")
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()









