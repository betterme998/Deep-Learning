import sys, os    # 导入系统相关模块，用于修改模块搜索路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 将父目录加入系统路径，以便导入同级或父级目录下的自定义模块
import numpy as np
from tqdm import tqdm                           # 导入进度条工具，用于显示训练循环的进度
import matplotlib.pyplot as plt                 # 导入绘图库，用于绘制准确率变化曲线
from dataset.mnist import load_mnist            # 导入 MNIST 数据集加载函数
from common.two_layer_net2 import TwoLayerNet   # 导入之前定义的两层神经网络类

# ------------------ 1. 加载并预处理数据 ------------------
# 加载 MNIST 数据集，返回训练数据和测试数据
# normalize=True ：将图像像素值从 0~255 归一化到 0.0~1.0
# one_hot_label=True ：将标签转换为 one-hot 编码（如数字 5 → [0,0,0,0,0,1,0,0,0,0]）

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# ------------------ 2. 创建网络实例 ------------------
# 输入层大小 784（28×28 像素），隐藏层 50 个神经元，输出层 10 个类别（数字 0-9）
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# ------------------ 3. 设定训练超参数 ------------------
iters_num = 10000               # 总迭代次数（参数更新次数）
train_size = x_train.shape[0]   # 训练样本总数（MNIST 为 60000）
batch_size = 100                # 每次迭代从训练集中随机抽取的样本数量（mini-batch 大小）
learning_rate = 0.1             # 学习率，控制参数更新的步长

# 用于记录训练过程中的损失值和准确率
train_loss_list = []            # 存储每个迭代的损失值
train_acc_list = []             # 存储每个 epoch 时的训练准确率
test_acc_list = []              # 存储每个 epoch 时的测试准确率

# 计算一个 epoch 对应的迭代次数（即遍历一次全部训练数据需要多少次 mini-batch 更新）
# max(..., 1) 防止批次大小大于训练集时出现除零或过小的 epoch 数
iter_per_epoch = max(train_size / batch_size, 1)

# ------------------ 4. 训练循环 ------------------
# 使用 tqdm 包装 range 以显示进度条
for i in tqdm(range(iters_num)):
  batch_mask = np.random.choice(train_size, batch_size)          # 从训练集中随机选取 batch_size 个样本的索引
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]                                  # 根据索引获取对应的输入数据和监督标签

  grad = network.gradient(x_batch, t_batch)                      # 通过反向传播计算当前批次上的梯度
  
  for key in ('W1', 'b1', 'W2', 'b2'):                           # 使用随机梯度下降（SGD）更新所有参数
    network.params[key] -= learning_rate * grad[key]             # 参数 = 参数 - 学习率 × 梯度

  loss = network.loss(x_batch, t_batch)                          # 计算当前批次上的损失值，并记录到列表中（用于后续观察损失变化趋势）
  train_loss_list.append(loss)

  if i % iter_per_epoch == 0:                                    # 每隔一个 epoch 的迭代次数，评估一次模型在训练集和测试集上的准确率
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(train_acc, test_acc)                                  # 打印当前 epoch 的训练准确率和测试准确率

# ------------------ 5. 绘制准确率变化曲线 ------------------
x = np.arange(len(train_acc_list))                              # 横坐标：epoch 数量（与 train_acc_list 长度相同）
plt.plot(x, train_acc_list, label='train acc')                  # 绘制训练准确率曲线，标记为圆形
plt.plot(x, test_acc_list, label='test acc', linestyle='--')    # 绘制测试准确率曲线，使用虚线，标记为方形
plt.xlabel("epochs")           # X 轴标签
plt.ylabel("accuracy")         # Y 轴标签
plt.ylim(0, 1.0)               # Y 轴范围设为 0~1（准确率范围）
plt.legend(loc='lower right')  # 图例放在右下角
plt.show()                     # 显示图形