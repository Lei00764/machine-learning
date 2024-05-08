"""
@File    :   Classifier.py
@Time    :   2024/05/08 19:12:41
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   Classifier task on iris dataset use logistic regression
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as SVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import matplotlib.pyplot as plt

# Step1: 加载鸢尾花数据集
iris = load_iris()
print(iris.data.shape)  # (150, 4)

# Step2: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.95, random_state=0
)


class MyModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def fit(self, X_train, y_train):
        if self.model_name == "LogisticRegression":
            self.model = LogisticRegression()
            self.model.fit(X_train, y_train)
        elif self.model_name == "SVM":
            self.model = SVM()
            self.model.fit(X_train, y_train)
        elif self.model_name == "KNN":
            self.model = KNeighborsClassifier()
            self.model.fit(X_train, y_train)
        elif self.model_name == "DecisionTree":
            self.model = DecisionTreeClassifier()
            self.model.fit(X_train, y_train)
        else:
            print("Model not supported!")

    def predict(self, X_test):
        if self.model_name == "KNN":
            print(self.model)
            print(X_test)

        return self.model.predict(X_test)


# Step3: 训练模型
# # You can change the model to LogisticRegression, SVM, KNN, DecisionTree
model = MyModel("KNN")
model.fit(X_train, y_train)

# Step4: 预测
y_pred = model.predict(X_test)

# Step5: 评估
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
precision = precision_score(y_test, y_pred, average="macro")
print("Precision: ", precision)
recall = recall_score(y_test, y_pred, average="macro")
print("Recall: ", recall)
f1 = f1_score(y_test, y_pred, average="macro")
print("F1: ", f1)

# Step6: 预测结果可视化 散点图
plt.figure(figsize=(12, 6))

plt.subplot(121)
# left: use Sepal length and Sepal width
# show the true label
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="r", label="0")
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="g", label="1")
plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], c="b", label="2")
# show the predict label
plt.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], c="r", marker="x", s=100)
plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], c="g", marker="x", s=100)
plt.scatter(X_test[y_pred == 2, 0], X_test[y_pred == 2, 1], c="b", marker="x", s=100)
plt.legend()
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris Classification by Sepal length and Sepal width")

# right: use Petal length and Petal width
# show the true label
plt.subplot(122)
plt.scatter(X_test[y_test == 0, 2], X_test[y_test == 0, 3], c="r", label="0")
plt.scatter(X_test[y_test == 1, 2], X_test[y_test == 1, 3], c="g", label="1")
plt.scatter(X_test[y_test == 2, 2], X_test[y_test == 2, 3], c="b", label="2")
# show the predict label
plt.scatter(X_test[y_pred == 0, 2], X_test[y_pred == 0, 3], c="r", marker="x", s=100)
plt.scatter(X_test[y_pred == 1, 2], X_test[y_pred == 1, 3], c="g", marker="x", s=100)
plt.scatter(X_test[y_pred == 2, 2], X_test[y_pred == 2, 3], c="b", marker="x", s=100)
plt.legend()
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.title("Iris Classification by Petal length and Petal width")
plt.show()
