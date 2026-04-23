"""
我们想用predict()函数一次性打包处理100张图像
x 的形状改为100 × 784，即100张图像的数据
实现：
输入数据的形状为 100 × 784
输出数据的形状为100 × 10

表示输入的100张图像的结果被一次性输出了

比如，x[0]和y[0]中保存了第0张图像及其推理结果

这种打包式的输入数据称为批（batch）
"""

# coding: utf-8
import sys # 访问命令行参数、系统路径以及Python运行时环境
import pickle  # 用于将数据快速化保存，方便快速加载序列化：
from pathlib import Path # Path 是一个面向对象的文件系统路径处理类
from typing import Any # 类型提示和类型注解的功能

import numpy as np # 科学计算机库，处理数组数据
import numpy.typing as npt # 导入 NumPy 的类型注解工具并起一个简短的别名，主要用于静态类型检查场景，让开发者能清晰地标注数组、标量、数据类型等对象的类型
# 1.使用pathlib 构建路径
# 将父目录添加到系统路径中
# Path(__file__).resolve()当前文件的绝对路径
# parent 获取父级目录

sys.path.append(str(Path(__file__).resolve().parent.parent))
print(str(Path(__file__).resolve().parent.parent))

from dataset.mnist import load_mnist # 从dataset包中导入MNIST数据集加载函数
from common.functions import sigmoid, softmax # 激活函数

def get_data() -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
  """
  获取经过预处理的MNIST测试数据.
  参数 normalize=True 表示将像素归一化到[0,1] 区间
  flatten=True 表示将 28x28 的图像展平为一维数组（784个元素）；
  one_hot_label=False 表示标签以原始数字形式(0~9) 存储，而非one-hot编码.[0,0,1,0,0,0,0,0,0,0] = 2
  """
  (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
  #本函数只返回测试图像，和测试标签
  return x_test, t_test

def init_network() -> dict[str, Any]:
  """
  从训练好的权重文件 "sample_weight.pkl" 中加载神经网络参数
  该文件是一个字典，包含各层的权重(W1, W2, W3)和偏置(b1, b2, b3)。
  """
  with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f) #从pickle格式的文件中读取数据并转换为Python的类型
  return network

def predict(network: dict[str, Any], x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
  """
  注：这里一次预测一张图像
  使用已加载的网络参数进行前向传播预测.
  network: 包含权重和偏置的字典
  x: 输入图像数据，形状为(784,)的一维数组
  返回：经过softmax层输出的10个类别的概率分别
  """
  # 从参数字典中取出各层权重和偏置
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  # 第一层：输入层 -> 隐藏层1
  a1 = np.dot(x, W1) + b1 # 主要用于矩阵的乘法运算，其中包括：向量内积、多维矩阵乘法、矩阵与向量的乘法。
  z1 = sigmoid(a1) # 激活函数sigmoid

  # 第二层：隐藏层1 -> 隐藏层2
  a2 = np.dot(z1, W2) + b2 #线性变换
  z2 = sigmoid(a2) # 激活函数sigmoid

  # 第三层：隐藏层2 -> 输出层
  a3 = np.dot(z2, W3) + b3 #线性变换
  y = softmax(a3) # 输出层使用softmax 将结果转换为概率分布
  return y

#------主程序-批处理-----
# 1.获取测试数据
x, t = get_data() # x: 测试图像数组(形状：10000x784),对应的正确标签(形状:10000,)
# 2.初始化网络（加载预训练权重）
network = init_network()

batch_size = 100 #批数量----------------------------------
# 3.统计预测正确的样本数
accuracy_cnt = 0

for i in range(0, len(x), batch_size): #------------------------批----
  x_batch = x[i:i+batch_size] # 获取当前批的测试图像
  y_batch = predict(network, x_batch) # 获取当前批的预测结果
  p = np.argmax(y_batch, axis=1) # 获取预测结果中的最大概率的索引,axis=1,沿着第1维方向（以第1维为轴）找到值最大的元素的索引（第0维对应第1个维度） 
  """
  例子：
  >>> x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6],... [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
  >>> y = np.argmax(x, axis=1)
  >>> print(y)
  [1 2 1 0]
  """
  accuracy_cnt += np.sum(p == t[i:i+batch_size])
  """
  例子
  >>> y = np.array([1, 2, 1, 0])
  >>> t = np.array([1, 2, 0, 0])
  >>> print(y==t)
  [True True False True]
  >>> np.sum(y==t)
  3
  """
  
#4. 计算并输出识别准确率
print(f"测试图像数组形状：{x.shape}") # (10000, 784)
print(f"第一个测试图像形状：{x[0].shape}") # (784,)
print(f"第一层权重形状：{network['W1'].shape}") # (784, 50) 
print(f"第二层权重形状：{network['W2'].shape}") # (50, 100)
print(f"第三层权重形状：{network['W3'].shape}") # (100, 10)
print("Accuracy:" + str(float(accuracy_cnt / len(x)))) #Accuracy:0.9352