"""
@File    :   OLS.py
@Time    :   2024/04/06 21:34:00
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   最小二乘法
"""

import numpy as np

# 输入数据
x = np.array([0, 3])
y = np.array([2, 1])


# 计算分子和分母
numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
denominator = np.sum((x - np.mean(x)) ** 2)

# 计算 w
w = numerator / denominator

# 计算 b
b = np.mean(y) - w * np.mean(x)

print("w =", w)
print("b =", b)

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

X = np.array([0, 2, 3]).reshape(-1, 1)
y = np.array([2, 2, 1])

loo = LeaveOneOut()
mse_scores = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

print(mse_scores)
mean_mse = np.mean(mse_scores)
print("Mean squared error (MSE) using Leave One Out cross-validation:", mean_mse)

import numpy as np

AD = np.array([42, 45, 49, 55, 57, 60, 62, 58, 54, 50, 44, 40])
FD = np.array([44, 46, 48, 50, 55, 60, 64, 60, 53, 48, 42, 38])

mae = np.mean(np.abs(AD - FD))
print("Mean Absolute Error (MAE):", mae)

mse = np.mean((AD - FD) ** 2)
print("Mean Squared Error (MSE):", mse)


import numpy as np

confusion_matrix = np.array([[40, 20, 10], [35, 85, 40], [0, 10, 20]])

TP = np.diag(confusion_matrix)
FN = np.sum(confusion_matrix, axis=1) - TP
FP = np.sum(confusion_matrix, axis=0) - TP
TN = np.sum(confusion_matrix) - (TP + FN + FP)

for i in range(len(TP)):
    print(f"For class {i+1}:")
    print("TP:", TP[i])
    print("FN:", FN[i])
    print("FP:", FP[i])
    print("TN:", TN[i])

precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

print("Precision for each class:", precision)
print("Recall for each class:", recall)

macro_precision = np.mean(precision)
macro_recall = np.mean(recall)

weighted_precision = np.sum(
    precision * np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
)
weighted_recall = np.sum(
    recall * np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
)

print("Macro-average Precision:", round(macro_precision, 4))
print("Macro-average Recall:", round(macro_recall, 4))
print("Weighted-average Precision:", round(weighted_precision, 4))
print("Weighted-average Recall:", round(weighted_recall, 4))
