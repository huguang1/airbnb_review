# 导入线性回归器算法模型
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
# 糖尿病数据集 ，训练一个回归模型来预测糖尿病进展
from sklearn import datasets

dia = datasets.load_diabetes()
# 提取特征数据和标签数据
data = dia.data[:, 2]
target = dia.target
# 训练样本和测试样本的分离，测试集20%
from sklearn.model_selection import train_test_split

features = np.column_stack((
    data,
))

# 创建线性回归模型
linear = LinearRegression()
# 用linear模型来训练数据:训练的过程是把x_train 和y_train带入公式W = (X^X)-1X^TY求出回归系数W
linear.fit(features, target)
# 对测试数据预测
# y_pre = linear.predict(x_test)
r2 = linear.score(features, target)
print(r2)
a = features[:, 0]
pccs = np.corrcoef(features[:, 0], target)
print(pccs)
# print(cross_val_score(linear, x_test, y_test, cv=10, scoring="r2").mean())
# n, p = x_test.shape
# adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
# print(adjusted_r2)
