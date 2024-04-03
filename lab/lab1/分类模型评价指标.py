"""
@Project ：lab1 
@File    ：分类模型评价指标.py
@Content ：混淆矩阵、准确率、精确率、召回率、F1值
@Author  ：Xiang Lei
@Email   ：xiang.lei.se@foxmail.com
@Date    ：4/2/2024 12:46 PM 
"""

from sklearn.metrics import confusion_matrix

y_true = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
y_predict = [0, 1, 0, 2, 1, 0, 1, 2, 0, 0, 1, 2]

matrix = confusion_matrix(y_true, y_predict)

"""
三类，横坐标是 Predicted，纵坐标是 Ground truth
矩阵中数字的含义：将 Ground truth 为 i 的样本预测为 j 的样本的个数
[[2 1 1]
 [1 2 1]
 [2 1 1]]
 
第 0 类的混淆矩阵
TP: 2 FN: 2
FP: 3 TN: 5

第 1 类的混淆矩阵
TP: 2 FN: 2
FP: 2 TN: 6
"""

# 计算 0 的 Precision and recall
Precision0 = 2 / (2 + 3)
Recall0 = 2 / (2 + 2)
F1_score0 = 2 * Precision0 * Recall0 / (Precision0 + Recall0)
print(Precision0, Recall0, F1_score0)
