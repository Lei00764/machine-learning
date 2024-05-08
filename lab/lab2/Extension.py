"""
@File    :   Extension.py
@Time    :   2024/05/08 19:44:10
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   mnist dataset classification task use SVM
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


import numpy as np
import matplotlib.pyplot as plt

# font setting New Roman
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

print(load_digits().data.shape)  # (1797, 64)

# Step1: 加载手写数字数据集
digits = load_digits()

# Step2: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=0
)

# Step3: 训练模型
model = SVM(probability=True)
model.fit(X_train, y_train)

# Step4: 预测
y_pred = model.predict(X_test)
y_pred_probility = model.predict_proba(X_test)
# print(y_pred_probility.shape)  # (360, 10)

# Step5: 评估
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="micro")
recall = recall_score(y_test, y_pred, average="micro")
f1 = f1_score(y_test, y_pred, average="micro")

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

# Step6: 可视化
# plt.figure(figsize=(10, 10))
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(X_test[i].reshape(8, 8), cmap="gray")
#     plt.title("True: %d, Predict: %d" % (y_test[i], y_pred[i]))
#     plt.axis("off")
# plt.show()

# Draw SVM loss, and the x axis is the value of gamma
gamma = np.logspace(-5, 4, 10)
train_loss = []
test_loss = []
for g in gamma:
    model = SVM(probability=True, gamma=g)
    model.fit(X_train, y_train)
    # 计算训练集上的loss
    y_train_pred = model.predict(X_train)
    train_loss.append(1 - accuracy_score(y_train, y_train_pred))
    # 计算测试集上的loss
    y_test_pred = model.predict(X_test)
    test_loss.append(1 - accuracy_score(y_test, y_test_pred))


# Draw bias-variance curve
test_loss_g = []
test_bias = []
test_variance = []
for g in gamma:
    model = SVM(probability=True, gamma=g)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_loss_g.append(1 - accuracy_score(y_test, y_pred))
    # compute bias
    # avg_y_pred = np.mean(y_pred)
    # avg_y_test = np.mean(y_test)
    # abs, then mean
    test_bias.append(np.mean(abs(y_test - y_pred)))
    # compute variance
    var_y_pred = np.mean((y_pred - y_test) ** 2)
    test_variance.append(var_y_pred)


# Draw Precision-Recall curve
classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)
precision_list = []
recall_list = []

for i in range(len(classes)):
    precision, recall, _ = precision_recall_curve(
        y_test_binarized[:, i], y_pred_probility[:, i]
    )
    precision_list.append(precision)
    recall_list.append(recall)


# Draw ROC curve
fpr_list = []
tpr_list = []
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_probility[:, i])
    fpr_list.append(fpr)
    tpr_list.append(tpr)

plt.figure(figsize=(14, 12))

# SVM Loss
plt.subplot(221)
plt.plot(gamma, train_loss, "b", label="Train Loss")
plt.plot(gamma, test_loss, "r", label="Test Loss")
plt.xscale("log")
plt.xlabel("gamma")
plt.ylabel("Loss")
plt.title("SVM Loss")
plt.legend()

# Bias-Variance
plt.subplot(222)
print(test_loss_g)
print(test_bias)
plt.plot(gamma, test_loss_g, "b", label="Test Loss")
plt.plot(gamma, test_bias, "r", label="Test Bias")
plt.plot(gamma, test_variance, "g", label="Test Variance")
plt.xscale("log")
plt.xlabel("gamma")
plt.ylabel("error")
plt.title("SVM Bias and Variance")
plt.legend()

# Precision-Recall curve
plt.subplot(223)
for i in range(len(classes)):
    plt.plot(recall_list[i], precision_list[i])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend(["Class " + str(i) for i in classes], loc="lower left")

# ROC curve
plt.subplot(224)
for i in range(len(classes)):
    plt.plot(fpr_list[i], tpr_list[i])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(["Class " + str(i) for i in classes], loc="lower right")

plt.show()
