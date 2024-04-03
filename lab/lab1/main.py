import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import warnings
from sklearn.exceptions import ConvergenceWarning

from Dataset_Partitioning import Hold_out, Bootstrapping
from Drawing import drawing_data, drawing_model, drawing_PR, drawing_ROC, drawing_models, drawing_PRs, drawing_ROCs
from Dataset import dataset
from Evaluation import get_Best_M

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 对M in Ms阶逻辑回归分类器，进行T次K折交叉验证，选择最优模型
# 绘制最佳模型在测试集上的ROC和PR图
# 绘制几个模型在测试集上的ROC和PR图

N = 2000  # 数据集的数量级
Ms = [1, 2, 5]  # 从M阶逻辑回归分类器中选择最优
T = 3
K = 5  # T次K折交叉验证

# 数据集生成
data = dataset(N)  # 可尝试生成不同的数据集
# drawing_data(data, '数据集分布')

# 数据集划分
# train_data, test_data = Hold_out(data, 0.5)  # 留出法
train_data, test_data = Bootstrapping(data, 2000)  # 自助法
print('训练集样本数量：' + str(len(train_data)))
print('测试集样本数量：' + str(len(test_data)))

# 定义M阶逻辑回归模型
# best_M = 2
# best_M = get_Best_M(train_data, Ms, 'Cross Validation', [T, K])
# best_M = get_Best_M(train_data, Ms, 'Hold Out', [0.3])
best_M = get_Best_M(train_data, Ms, 'Bootstrapping', [1000])
best_model = make_pipeline(PolynomialFeatures(degree=best_M), LogisticRegression())
# 在完整train_data上训练最佳模型
best_model.fit(train_data[:, 1:], train_data[:, 0])  # 可自行设置更多参数
"""
train_data 第一列为标签，后面的列为特征
"""
# 结果
output = [q for p, q in best_model.predict_proba(test_data[:, 1:])]
# output取0~1，设定阈值，将其分类0、1两类

# 评价
boolnum = len(test_data)
boolT = 0
for i in range(boolnum):
    if test_data[i][0] == 0:
        if output[i] < 0.5:  # 这里以0.5为分界进行分类
            boolT += 1
    elif test_data[i][0] == 1:
        if output[i] >= 0.5:
            boolT += 1

Accuracy = boolT / boolnum  # 可尝试其他评价指标
print(str(best_M) + '阶逻辑回归分类器在测试集上的准确率：' + str(round(100 * Accuracy, 2)) + '%')

# 绘图
# 最佳模型：
# drawing_model(train_data, best_model, str(best_M) + '阶逻辑回归模型在训练集上的结果')
# drawing_model(test_data, best_model, str(best_M) + '阶逻辑回归模型在测试集上的结果')
# drawing_PR(test_data[:, 0], output, str(best_M) + '阶模型的PR曲线图')
# drawing_ROC(test_data[:, 0], output, str(best_M) + '阶模型的ROC曲线图')

# 所有模型：
# models = []
# outputs = []
# for M in Ms:
#     M_model = make_pipeline(PolynomialFeatures(degree=M), LogisticRegression())
#     M_model.fit(train_data[:, 1:], train_data[:, 0])
#     models.append(M_model)
#     output = [q for p, q in M_model.predict_proba(test_data[:, 1:])]
#     outputs.append(output)
# drawing_models(models, test_data, Ms, '各阶逻辑回归分类器在测试集上的表现')
# drawing_PRs(outputs, test_data, Ms, '各阶逻辑回归分类器PR曲线')
# drawing_ROCs(outputs, test_data, Ms, '各阶逻辑回归分类器ROC曲线')
plt.show()
