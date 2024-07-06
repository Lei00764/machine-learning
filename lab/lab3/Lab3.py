import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.models import Sequential  # 采用贯序模型
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, Flatten
from keras.models import Model

# from keras.optimizer_v2.gradient_descent import SGD
from keras.optimizers import SGD
from keras.datasets import mnist

# from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

# from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import plot_model

# 固定随机数种子，提高结果可重复性（在CPU上测试有效）
tf.random.set_seed(233)
np.random.seed(233)

"""第一步：选择模型"""
model = Sequential()  # 采用贯序模型

print(tf.test.is_gpu_available())
"""第二步：构建网络层"""
# 在此处构建你的网络
#####################################################################################
model.add(Input((28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(Flatten())

model.add(Dense(500))
model.add(Dropout(0.5))  # 随机失活，缓解过拟合
model.add(Activation("relu"))
model.add(Dense(128))  # 添加 128 个神经元的全连接层
model.add(Dropout(0.5))
model.add(Activation("relu"))
model.add(Dense(10))  # 输出层，10个神经元，对应0-9十个数字
model.add(Activation("softmax"))
#####################################################################################

plot_model(
    model, to_file="model.png", show_shapes=True, show_layer_names=False, rankdir="TB"
)

"""第三步：网络优化/编译/模型输出"""
# 在此处调整优化器
# learning_rate：大于0的浮点数，学习率
# momentum：大于0的浮点数，动量参数
# decay：大于0的浮点数，每次更新后的学习率衰减值
# nesterov：布尔值，确定是否使用Nesterov动量
sgd = SGD(
    learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True
)  # 优化函数，设定学习率（lr）等参数

# 在此处调整损失函数，并编译网络
model.compile(
    loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
)  # 使用交叉熵作为loss函数

# 在此处输出网络的架构。此处参数可以不用调整。
# model表示自定义的模型名 to_file表示存储的文件名 show_shapes是否显示形状  rankdir表示方向T(top)B(Bottow)
# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB')

"""第四步：训练"""

# 数据集获取 mnist 数据集的介绍可以参考 https://blog.csdn.net/simple_the_best/article/details/75267863
(X_train, y_train), (X_test, y_test) = (
    mnist.load_data()
)  # 使用Keras自带的mnist工具读取数据（第一次运行需要联网）

# 数据处理与归一化
# 注意：X_train和X_test可以直接输入卷积层，但需要先Flatten才能输入全连接层
X_train = X_train.reshape((60000, 28, 28, 1)).astype("float") / 255
X_test = X_test.reshape((10000, 28, 28, 1)).astype("float") / 255

# 生成OneHot向量
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

# 在此处调整训练细节
"""
   .fit的一些参数
   batch_size：对总的样本数进行分组，每组包含的样本数量
   epochs ：训练次数
   shuffle：是否把数据随机打乱之后再进行训练
   validation_split：拿出百分之多少用来做交叉验证
   verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
"""
history = model.fit(
    X_train,
    Y_train,
    batch_size=128,
    epochs=50,
    shuffle=True,
    verbose=2,
    validation_split=0.3,
)

"""第五步：输出与可视化"""
print("test set")
# 误差评价 ：按batch计算在batch用到的输入数据上模型的误差，并输出测试集准确率
scores = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
print("The test loss is %f" % scores[0])
print("The accuracy of the model is %f" % scores[1])

# 在此处实现你的可视化功能
#####################################################################################
# loss 曲线 and accuracy 曲线
plt.figure(1)
plt.plot(np.arange(1, 51), history.history["loss"], label="train_loss")
plt.plot(np.arange(1, 51), history.history["val_loss"], label="val_loss")
plt.title("loss")
plt.legend()
plt.savefig("loss.png")

plt.figure(2)
plt.plot(np.arange(1, 51), history.history["accuracy"], label="tarin_accuracy")
plt.plot(np.arange(1, 51), history.history["val_accuracy"], label="val_accuracy")
plt.title("accuracy")
plt.legend()
plt.savefig("accuracy.png")
#####################################################################################
