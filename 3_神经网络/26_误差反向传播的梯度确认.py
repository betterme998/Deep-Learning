"""
在确认误差反向传播法的实现是否正确时，是需要用到数值微分的。  
确认数值微分求出的梯度结果和误差反向传播法求出的结果是否一致（严格地讲，是  
非常相近）的操作称为梯度确认（gradient check）。
"""
import sys, os  # 导入系统与路径处理模块                    
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 将父目录加入系统路径，确保可以导入自定义的 common 模块
import numpy as np
from dataset.mnist import load_mnist # 导入 MNIST 数据集的加载函数
from common.two_layer_net2 import TwoLayerNet # 导入前面定义的两层神经网络类

# 加载 MNIST 数据集，并进行归一化和 one-hot 编码
# normalize=True ：将像素值归一化到 0.0 ~ 1.0
# one_hot_label=True ：将标签转换为 one-hot 编码（如数字 5 变为 [0,0,0,0,0,1,0,0,0,0]）
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) 

# 创建网络实例：输入 784 个神经元（28×28图像），隐藏层 50 个神经元，输出 10 个类别
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 取训练数据的前 3 个样本作为一个迷你批次，用于梯度检查
x_batch = x_train[:3]
t_batch = t_train[:3]

# 使用数值微分方法计算梯度（速度慢，但实现简单，用作基准）
grad_numerical = network.numerical_gradient(x_batch, t_batch)
# 使用反向传播方法计算梯度（速度快，实际训练中使用）
grad_backprop = network.gradient(x_batch, t_batch)

# 逐参数比较两种方法得到的梯度差异
# 差异用平均绝对误差（MAE）衡量，值越小说明反向传播实现越正确
for key in grad_numerical.keys():
  # np.abs() 是 NumPy 库中的一个函数，用于计算数组中每个元素的绝对值
  diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
  print(key + ":" + str(diff))