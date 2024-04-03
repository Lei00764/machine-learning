import random
import numpy as np
import math


# 由于是二分类问题，所以这里默认数据集label只有0和1，大家也可以挑战更高难度（未知标签情况）

# fold折交叉验证法的第k次（即取fold份数据中的第k份作为测试集,k从1开始）
def Cross_Validation(data, fold, k):
    # 分层采样，将数据集分为fold份
    class0 = []  # 正
    class1 = []  # 负
    for d in data:
        if d[0] == 0:
            class0.append(d)
        else:
            class1.append(d)

    # 交叉验证
    training_set = []
    validation_set = []
    for i in range(len(class0)):
        if i % fold == k:
            validation_set.append(class0[i])
        else:
            training_set.append(class0[i])
    for i in range(len(class1)):
        if i % fold == k:
            validation_set.append(class1[i])
        else:
            training_set.append(class1[i])

    # 随机打乱
    random.shuffle(training_set)
    random.shuffle(validation_set)
    # 转成numpy数组
    training_set = np.array(training_set)
    validation_set = np.array(validation_set)

    return training_set, validation_set


# 测试样本占比为test_ratio的留出法
def Hold_out(data, test_ratio):
    class0 = []
    class1 = []

    # 验证集划分
    for d in data:
        if d[0] == 0:
            class0.append(d)
        else:
            class1.append(d)
    train_data = []
    test_data = []

    # 验证集划分
    for i in range(len(class0)):
        if i < len(class0) * test_ratio:
            test_data.append(class0[i])
        else:
            train_data.append(class0[i])

    for i in range(len(class1)):
        if i < len(class1) * test_ratio:
            test_data.append(class1[i])
        else:
            train_data.append(class1[i])

    return np.array(train_data), np.array(test_data)


# 训练样本抽样times次的自助法
def Bootstrapping(data, times):
    if len(data) < times:
        times = len(data) * 3 / 4

    test_data = []  # 未出现在 train_data 中的数据
    selected_index = []
    for i in range(times):
        selected_index.append(random.randint(0, len(data) - 1))

    train_data = data[selected_index]

    # 不在 selected_index 中的数据作为测试集
    for i in range(len(data)):
        if i not in selected_index:
            test_data.append(data[i])

    return np.array(train_data), np.array(test_data)
