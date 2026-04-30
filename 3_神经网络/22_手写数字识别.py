"""
2层神经网络（隐藏层为1层的网络）为对象，使用MNIST数据集进行学习。
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))# 将父目录加入路径，以便导入common文件夹中的模块
from common.functions import * # 导入激活函数、损失函数等
from common.gradient import numerical_gradient #导入数值梯度计算函数
import numpy as np

class TwoLayerNet:
  """两层全连接神经网络（输入层-隐藏层-输出层）"""

  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    """
    初始化网络权重和偏置
    参数：
      input_size: 输入层神经元个数（特征数）
      hidden_size: 隐藏层神经元个数
      output_size: 输出层神经元个数（类别数）
      weight_init_std: 权重初始化的标准差，默认0.01
    """
    self.params = {}
    # W1: 输入层到隐藏层权重，形状为 (input_size, hidden_size)
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    # b1:隐藏层偏置，形状为 (hidden_size,)并用零填充
    self.params['b1'] = np.zeros(hidden_size)
    # W2: 隐藏层到输出层权重，形状为 (hidden_size, output_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    # b2: 输出层偏置，形状为 (output_size,)
    self.params['b2'] = np.zeros(output_size)

  def predict(self, x):
    """
    前向传播，计算输出预测值
    参数：
      x: 输入数据，形状为（batch_size, input_size）(如一百张照片，照片大小为28*28)
    返回：
      y: softmax输出概率，形状为（barch_size, output_size）(如一百张照片，每个照片有十种分类0-9)
    """
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    a1 = np.dot(x, W1) + b1  # 隐藏层线性变换
    z1 = sigmoid(a1)         # 隐藏层激活
    a2 = np.dot(z1, W2) + b2 # 输出层线性变换
    y = softmax(a2)          # 输出层激活，得到各类别概率

    return y
  
  def loss(self, x, t):
    """
    计算交叉熵损失
    参数：
      x: 输入数据，形状为（batch_size, input_size）
      t: 监督标签 （one-hot编码）, 形状为（batch_size, output_size）
    返回：
      交叉熵损失值（标量）
    """
    y = self.predict(x)
    return cross_entropy_error(y, t)
  
  def accuracy(self, x, t):
    """
    计算分类准确率
    参数：
      x: 输入数据，形状为（batch_size, input_size）
      t: 监督标签（one-hot编码），形状为（batch_size, output_size）
    返回：
      准确率（0.0~1.0）
    """
    y = self.predict(x)
    y = np.argmax(y, axis=1) #预测类别
    t = np.argmax(t, axis=1) #真实类别，one-hot 向量转回原始的类别标签

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy
  
  def numerical_gradient(self, x, t):
    """
    使用数值微分计算梯度（速度慢，主要用于梯度准确性检查）
    参数：
      x:输入数据
      t:监督标签
    返回：
      grads：包含各参数梯度的字典
    """

    #定义以损失函数值为输出的函数，用于数值微分
    loss_W = lambda W: self.loss(x, t)

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads
  
  def gradient(self, x, t):
    """
    使用误差反向传播算法高速计算梯度
    参数：
      x:输入数据
      t:监督标签（one-hot）
  返回:
    grads:包含各参数梯度的字典
    """
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']
    grads = {}

    batch_num = x.shape[0]

    #----- 前向传播（与predict相同）----
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    # ----- 反向传播 -----
    # softmax与交叉熵误差的联合梯度简化为 (y - t) / batch_num
    dy = (y - t) / batch_num
    # 输出层权重梯度：W2的梯度 = 隐藏层输出转置 * 上游梯度
    grads['W2'] = np.dot(z1.T, dy) # 所以 z1.T 就是通过转置，让前一层激活值与当前层梯度在样本维度上正确对齐，从而一次性算出所有权重的累积梯度。
    # 输出层偏置梯度：b2的梯度 = 上游梯度按样本求和
    grads['b2'] = np.sum(dy, axis=0)

    # 向隐藏层反向传播
    dz1 = np.dot(dy, W2.T)    ## 传到sigmoid前的梯度
    da1 = sigmoid_grad(a1) * dz1