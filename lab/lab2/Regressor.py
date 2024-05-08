"""
@File    :   Regressor.py
@Time    :   2024/05/08 19:05:42
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   Regressor tast on diabetes dataset use linear regression
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt

# Step1: 加载糖尿病数据集
diabetes = load_diabetes()
# print(diabetes.data.shape)  # (442, 10)

# Step2: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=0
)

# Step3: 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# Step4: 预测
y_pred = model.predict(X_test)

# Step5: 评估
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

# Step6: 输出模型参数
print("Weights: ", model.coef_)  # 权重
print("Intercept: ", model.intercept_)  # 截距

# Step7: 预测结果可视化
plt.figure()
plt.plot(range(len(y_test)), y_test, "r", label="True")
plt.plot(range(len(y_pred)), y_pred, "b", label="Predict")
plt.legend()
plt.show()
