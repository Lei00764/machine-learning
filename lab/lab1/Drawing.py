import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score

# matplotlib画图中中文显示会有问题，需要这两行设置默认字体可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def drawing_data(data, title):
    Label = []
    Input1 = []
    Input2 = []

    for d in data:
        Label.append(d[0])
        Input1.append(d[1])
        Input2.append(d[2])

    plt.figure(title)

    for i in range(len(Label)):
        if Label[i] == 1:
            plt.scatter(Input1[i], Input2[i], c='r', marker='+')
        else:
            plt.scatter(Input1[i], Input2[i], c='b', marker='.')

    plt.title(title)
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    plt.xlim(-8, 6)
    plt.ylim(-8, 6)


def drawing_model(data, model, title):
    Label = []
    Input1 = []
    Input2 = []

    for d in data:
        Label.append(d[0])
        Input1.append(d[1])
        Input2.append(d[2])

    plt.figure(title)
    xx, yy = np.meshgrid(np.arange(-8, 6, 0.01),
                         np.arange(-8, 6, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    # 　plt.contour(xx, yy, Z, colors='k', linewidths=1.5)

    for i in range(len(Label)):
        if Label[i] == 1:
            plt.scatter(Input1[i], Input2[i], c='r', marker='+')
        else:
            plt.scatter(Input1[i], Input2[i], c='b', marker='.')

    plt.title(title)
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    plt.xlim(-8, 6)
    plt.ylim(-8, 6)


def drawing_models(models, test_data, Ms, title):
    Label = []
    Input1 = []
    Input2 = []

    for d in test_data:
        Label.append(d[0])
        Input1.append(d[1])
        Input2.append(d[2])

    plt.figure(title)

    for i in range(len(Label)):
        if Label[i] == 1:
            plt.scatter(Input1[i], Input2[i], c='r', marker='+')
        else:
            plt.scatter(Input1[i], Input2[i], c='b', marker='.')

    xx, yy = np.meshgrid(np.arange(-8, 6, 0.01),
                         np.arange(-8, 6, 0.01))
    for M_model in models:
        Z = M_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.contour(xx, yy, Z, colors='k', linewidths=1.5)

    plt.title(title)
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    plt.xlim(-8, 6)
    plt.ylim(-8, 6)


"""
    设0为True，1为false
        predict
    gt  0       1
    0   TP      FN
    1   FP      TN

"""


def drawing_PR(Label, Output, title):
    # 对 Output 排序，对应 Label 也按照这个顺序排列，逆序
    sorted_index = np.argsort(Output)[::-1]
    Label = np.array(Label)[sorted_index]
    Output = np.array(Output)[sorted_index]

    precision_list = []
    recall_list = []

    for i in range(len(Label)):
        # 将阈值设置成当前的 Output[i]
        threshold = Output[i]
        # >= threshold --- 1; < threshold --- 0
        new_output = [1 if x >= threshold else 0 for x in Output]
        TP, FN, FP, TN = 0, 0, 0, 0
        for i in range(len(Label)):
            if Label[i] == 1 and new_output[i] == 1:
                TP += 1
            elif Label[i] == 1 and new_output[i] == 0:
                FN += 1
            elif Label[i] == 0 and new_output[i] == 1:
                FP += 1
            else:
                TN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        precision_list.append(precision)
        recall_list.append(recall)
    plt.figure(title)
    plt.plot(recall_list, precision_list)
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(['PR曲线'], loc='lower left')


def drawing_ROC(Label, Output, title):
    # 对 Output 排序，对应 Label 也按照这个顺序排列，逆序
    sorted_index = np.argsort(Output)[::-1]
    Label = np.array(Label)[sorted_index]
    Output = np.array(Output)[sorted_index]

    tpr_list = []
    fpr_list = []

    for i in range(len(Label)):
        # 将阈值设置成当前的 Output[i]
        threshold = Output[i]
        # >= threshold --- 1; < threshold --- 0
        new_output = [1 if x >= threshold else 0 for x in Output]
        TP, FN, FP, TN = 0, 0, 0, 0
        for i in range(len(Label)):
            if Label[i] == 1 and new_output[i] == 1:
                TP += 1
            elif Label[i] == 1 and new_output[i] == 0:
                FN += 1
            elif Label[i] == 0 and new_output[i] == 1:
                FP += 1
            else:
                TN += 1
        tpr = TP / (TP + FN)
        fpr = FP / (TN + FP)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    plt.figure(title)
    plt.plot(fpr_list, tpr_list)
    plt.title(title)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(['ROC曲线'], loc='lower right')


def drawing_PRs(outputs, test_data, Ms, title):
    for Output in outputs:
        # 对 Output 排序，对应 Label 也按照这个顺序排列，逆序
        sorted_index = np.argsort(Output)[::-1]
        Label = test_data[:, 0]
        Label = np.array(Label)[sorted_index]
        Output = np.array(Output)[sorted_index]

        precision_list = []
        recall_list = []

        for i in range(len(Label)):
            # 将阈值设置成当前的 Output[i]
            threshold = Output[i]
            # >= threshold --- 1; < threshold --- 0
            new_output = [1 if x >= threshold else 0 for x in Output]
            TP, FN, FP, TN = 0, 0, 0, 0
            for i in range(len(Label)):
                if Label[i] == 1 and new_output[i] == 1:
                    TP += 1
                elif Label[i] == 1 and new_output[i] == 0:
                    FN += 1
                elif Label[i] == 0 and new_output[i] == 1:
                    FP += 1
                else:
                    TN += 1
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            precision_list.append(precision)
            recall_list.append(recall)
        plt.figure(title)
        plt.plot(recall_list, precision_list)
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend([str(M) + ' 阶逻辑回归分类器 PR 曲线' for M in Ms], loc='lower left')


def drawing_ROCs(outputs, test_data, Ms, title):
    for Output in outputs:
        # 对 Output 排序，对应 Label 也按照这个顺序排列，逆序
        sorted_index = np.argsort(Output)[::-1]
        Label = test_data[:, 0]
        Label = np.array(Label)[sorted_index]
        Output = np.array(Output)[sorted_index]

        tpr_list = []
        fpr_list = []

        for i in range(len(Label)):
            # 将阈值设置成当前的 Output[i]
            threshold = Output[i]
            # >= threshold --- 1; < threshold --- 0
            new_output = [1 if x >= threshold else 0 for x in Output]
            TP, FN, FP, TN = 0, 0, 0, 0
            for i in range(len(Label)):
                if Label[i] == 1 and new_output[i] == 1:
                    TP += 1
                elif Label[i] == 1 and new_output[i] == 0:
                    FN += 1
                elif Label[i] == 0 and new_output[i] == 1:
                    FP += 1
                else:
                    TN += 1
            tpr = TP / (TP + FN)
            fpr = FP / (TN + FP)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        plt.figure(title)
        plt.plot(fpr_list, tpr_list)
    plt.title(title)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend([str(M) + ' 阶逻辑回归分类器 ROC 曲线' for M in Ms], loc='lower right')
