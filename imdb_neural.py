import random

from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras import models
from keras import layers
import numpy as np

# 仅保留训练数据中前10000个最常出现的单词，舍弃低频单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)  # 向量化训练数据
x_test = vectorize_sequences(test_data)  # 向量化测试数据
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
for i, v in enumerate(y_train):
    if v == 1:
        y_train[i] = random.random()*5
for i, v in enumerate(y_test):
    if v == 1:
        y_test[i] = random.random()*5

# 模型由两个中间层（每层16个隐藏单元）和一个输出层（输出标量）组成
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=1, activation='linear'))

# 编译模型，使用内置的优化器、损失函数
model.compile(optimizer='adam', loss='mse')

# 验证数据
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 训练模型，每批次512条样本，共20次迭代，把训练过程中的数据保存在history
history = model.fit(partial_x_train, partial_y_train, epochs=80, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

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
