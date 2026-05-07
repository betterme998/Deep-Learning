"""
mini-batch版2层神经网络（隐藏层为1层的网络）为对象，使用MNIST数据集进行学习。
就是从训练数据中随机选择一部分数据（称为mini-batch），再以这些mini-batch为对象，使用梯度法更新参数的过程。

，mini-batch的大小为100，需要每次从60000个训练数据中随机
取出100个数据（图像数据和正确解标签数据）。然后，对这个包含100笔数
据的mini-batch求梯度，使用随机梯度下降法（SGD）更新参数。这里，梯
度法的更新次数（循环的次数）为10000。每更新一次，都对训练数据计算损
失函数的值，并把该值添加到数组中。用图像来表示这个损失函数的值的推
移
"""
import sys, os
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))# 将父目录加入路径，以便导入common文件夹中的模块
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist  # 用于加载MNIST数据集
from common.two_layer_net import TwoLayerNet # 自定义的两层神经网络类

# 加载MNIST数据集，并进行标准化和one-hot标签处理
# normalize=True: 将像素值归一化到0~1；one_hot_label=True: 标签转换为one-hot向量
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 初始化神经网络：输入层784个神经元（28x28），隐藏层50个神经元，输出层10个神经元（对应0~9）
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#  超参数设置
iters_num = 10000             # 总迭代次数（可自行调整）
train_size = x_train.shape[0] # 训练样本数量
batch_size = 100              # 每次迭代使用的批量大小
learning_rate = 0.1           # 学习率

# 用于记录训练过程中的损失值和准确率
train_loss_list = []          # 每次迭代的训练损失
train_acc_list = []           # 每个epoch的训练准确率
test_acc_list = []            # 每个epoch的测试准确率

# 计算每个epoch对应的迭代次数（一个epoch定义为完整遍历一次训练集所需的迭代次数）
iter_per_epoch = max(train_size // batch_size, 1)

# 开始训练循环
for i in tqdm(range(iters_num)):# 获取mini-atch
  # 从训练集中随机选取batch_size个样本（Mini-batch学习）
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask] # 批量输入数据
  t_batch = t_train[batch_mask] # 批量标签

  # 计算梯度（此处使用误差反向传播，若使用数值微分可改为numerical_gradient）
  # grad = network.numerical_gradient(x_batch, t_batch)
  grad = network.gradient(x_batch, t_batch)

  # 利用梯度下降法更新网络参数（权重和偏置）
  for key in ('W1', 'b1', 'W2', 'b2'):
    network.params[key] -= learning_rate * grad[key]

  # 计算当前批量的损失，并记录下来
  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)

  # 每个epoch结束时，计算并记录训练集合和测试集的准确率
  if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)  # 使用整个训练集计算epoch准确率
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    # 输出当前epoch的准确率
    print("train acc, test acc | " + str(train_acc) + "," + str(test_acc))


# 绘制准确率随epoch变化的曲线
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list)) # # epoch序号作为横轴
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)                # 纵轴范围0~1
plt.legend(loc='lower right')   # 图例放在右下角
plt.show()