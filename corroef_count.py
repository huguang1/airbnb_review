import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

result = []
# 使用numpy计算数据特征和标签的相关系数
for i in range(np.shape(iris.data)[1]):
    pccs = np.corrcoef(iris.data[:, i], iris.target)
    print(pccs)
    # result.append(pccs[:, 1][0])

# print(result)
# # 对列表中的数都保留两位小数
# result1 = []
# for i in range(len(result)):
#     result1.append(round(result[i], 3))



