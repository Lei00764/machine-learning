import numpy as np
from Dataset_Partitioning import Cross_Validation, Hold_out, Bootstrapping
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def get_Best_M(train_data, Ms, method, parameters):
    if method == 'Cross Validation':
        T = parameters[0]
        K = parameters[1]
        return get_CV(train_data, Ms, T, K)
    elif method == 'Hold Out':
        test_ratio = parameters[0]
        return get_HO(train_data, Ms, test_ratio)
    elif method == 'Bootstrapping':
        times = parameters[0]
        return get_B(train_data, Ms, times)


def get_CV(train_data, Ms, T, K):
    """
    K折交叉验证
    :param train_data:
    :param Ms:
    :param T:
    :param K:
    :return:
    """
    Best_M = 0
    best_acc = 0
    flag = 0
    # 先选择参数，在进行交叉验证
    for M in Ms:
        model = make_pipeline(PolynomialFeatures(degree=M), LogisticRegression())

        for i in range(T):
            # 交叉验证，使用分层采用
            training_set, validation_set = Cross_Validation(train_data, K, T)
            if flag == 0:
                flag = 1
                print("交叉验证，对训练集进一步划分，结果如下：")
                print("训练集样本数量：" + str(len(training_set)))
                print("验证集样本数量：" + str(len(validation_set)))
            # 训练模型
            model.fit(training_set[:, 1:], training_set[:, 0])
            # 验证模型
            # 验证集上的输出
            output = [q for p, q in model.predict_proba(validation_set[:, 1:])]
            # 评价
            boolnum = len(validation_set)
            boolT = 0
            for j in range(boolnum):
                if validation_set[j][0] == 0:
                    if output[j] < 0.5:
                        boolT += 1
                elif validation_set[j][0] == 1:
                    if output[j] >= 0.5:
                        boolT += 1
            acc = boolT / boolnum
            print(str(M) + '阶逻辑回归模型' + str(T) + '次' + str(K) + '折交叉检验的平均准确率为' + str(
                round(100 * acc, 2)) + '%')
            if acc > best_acc:
                best_acc = acc
                Best_M = M
    print("最佳模型为" + str(Best_M) + "阶逻辑回归模型，其在交叉验证集上平均准确率为" + str(
        round(100 * best_acc, 2)) + "%")
    return Best_M


def get_HO(train_data, Ms, test_ratio):
    """
    留出法
    :param train_data:
    :param Ms:
    :param test_ratio:
    :return:
    """
    Best_M = 0
    best_acc = 0
    train_data, validation_data = Hold_out(train_data, test_ratio)

    for M in Ms:
        model = make_pipeline(PolynomialFeatures(degree=M), LogisticRegression())
        model.fit(train_data[:, 1:], train_data[:, 0])
        output = [q for p, q in model.predict_proba(validation_data[:, 1:])]
        boolnum = len(validation_data)
        boolT = 0
        for j in range(boolnum):
            if validation_data[j][0] == 0:
                if output[j] < 0.5:
                    boolT += 1
            elif validation_data[j][0] == 1:
                if output[j] >= 0.5:
                    boolT += 1
        acc = boolT / boolnum
        print(str(M) + '阶逻辑回归模型在验证集上的准确率为' + str(round(100 * acc, 2)) + '%')
        if acc > best_acc:
            best_acc = acc
            Best_M = M

    print("最佳模型为" + str(Best_M) + "阶逻辑回归模型，其在验证集上准确率为" + str(round(100 * best_acc, 2)) + "%")
    return Best_M


def get_B(train_data, Ms, times):
    """
    自助法
    :param train_data:
    :param Ms:
    :param times:
    :return:
    """
    Best_M = 2
    best_acc = 0
    train_data, validation_data = Bootstrapping(train_data, times)

    for M in Ms:
        model = make_pipeline(PolynomialFeatures(degree=M), LogisticRegression())
        model.fit(train_data[:, 1:], train_data[:, 0])
        output = [q for p, q in model.predict_proba(validation_data[:, 1:])]
        boolnum = len(validation_data)
        boolT = 0
        for j in range(boolnum):
            if validation_data[j][0] == 0:
                if output[j] < 0.5:
                    boolT += 1
            elif validation_data[j][0] == 1:
                if output[j] >= 0.5:
                    boolT += 1
        acc = boolT / boolnum
        print(str(M) + '阶逻辑回归模型在验证集上的准确率为' + str(round(100 * acc, 2)) + '%')
        if acc > best_acc:
            best_acc = acc
            Best_M = M

    print("最佳模型为" + str(Best_M) + "阶逻辑回归模型，其在验证集上准确率为" + str(round(100 * best_acc, 2)) + "%")
    return Best_M
