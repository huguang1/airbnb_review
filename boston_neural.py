import numpy as np
import matplotlib.pyplot as plt
from tensorflow import random
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

from keras.datasets import boston_housing

(train_x, train_y), (test_x, test_y) = boston_housing.load_data()

np.random.seed(1)
random.set_seed(1)

for k in [5, 20, 50]:
    model = Sequential()
    model.add(BatchNormalization(input_dim=13))
    model.add(Dense(k,
                    kernel_initializer='random_uniform',
                    activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                    use_bias=True))
    model.add(Dropout(0.1))
    model.add(Dense(1, use_bias=True))

model.compile(optimizer='adam', loss='mse')

history = model.fit(train_x,
                    train_y,
                    epochs=500,
                    batch_size=50,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=20)], verbose=False)
# print(min(history.history['val_1oss']))
model.summa
# plt.plot(history.history['loss'], c='blue')
# plt.plot(history.history['val_loss'], C='red')
# plt.show()

pred_y = model.predict(test_x)[:, 0]

print(mean_squared_error(test_y, pred_y))


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 4), dpi=80)
plt.plot(range(len(test_y)), test_y, ls='-.', LW=2, C='r', label="真实值")
plt.plot(range(len(pred_y)), pred_y, ls='-', LW=2, C="b", labe1="预测值")
plt.grid(alpha=0.4, linestyle=':')
plt.legend()
plt.xlabel('number')
plt.ylabel('房价')

plt.show()
