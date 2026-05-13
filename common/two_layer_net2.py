"""
使用了激活函数层来计算
2层神经网络（隐藏层为1层的网络）为对象，使用MNIST数据集进行学习。
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))# 将父目录加入路径，以便导入common文件夹中的模块
import numpy as np
from common.layers import *                                   # 导入所有层类（Affine, Relu, SoftmaxWithLoss等）
from common.gradient import numerical_gradient                # 导入数值梯度计算函数
from collections import OrderedDict                           # OrderedDict 是一种能记住键插入顺序的字典。

class TwoLayerNet:
  """
  两层神经网络（一个隐藏层 + 一个输出层）
  输入层 → Affine → ReLU → Affine → SoftmaxWithLoss
  """
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    """
    初始化网络的结构和参数

    参数:
        input_size  : 输入层神经元数量（例如MNIST为784）
        hidden_size : 隐藏层神经元数量
        output_size : 输出层神经元数量（例如MNIST为10）
        weight_init_std : 权重初始化的标准差，默认为0.01
    """
    # ---------- 初始化权重和偏置 ----------
    # 使用正态分布随机数初始化权重，并乘以 weight_init_std 控制初始值的尺度
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)                                       ## 偏置初始化为0
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

    # ---------- 生成网络层 ----------
    # 使用OrderedDict以保证前向/反向传播时层的顺序固定（插入顺序即执行顺序）
    self.layers = OrderedDict()
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])  # 第一层全连接
    self.layers['Relu1'] = Relu()                                          # ReLU激活函数
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])  # 第二层全连接

    # 最后一层独立保存，因为它的功能（Softmax + 损失计算）与普通层不同
    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x):
    """正向传播，只进行预测（不计算损失），返回各层的原始得分（未经过Softmax）"""
    for layer in self.layers.values():
      x = layer.forward(x) # 依次通过每一层的前向计算
    
    return x
  
  # x: 输入数据, t:监督数据
  def loss(self, x, t):
    """
    计算损失值

    参数:
        x : 输入数据（mini-batch）
        t : 监督数据（标签）
    返回:
        交叉熵损失值（标量）
    """
    y = self.predict(x)                              # 前向传播得到网络输出（得分）
    return self.lastLayer.forward(y, t)              # SoftmaxWithLoss内部完成Softmax + 交叉熵计算
  
  def accuracy(self, x, t):
    """
    计算识别精度（准确率）

    参数:
        x : 输入数据
        t : 监督数据（one-hot或标签索引）
    返回:
        准确率（0.0 ~ 1.0）
    """
    y = self.predict(x)                           # 得到输出得分
    y = np.argmax(y, axis=1)                      # 取每个样本得分最大的索引作为预测类别
    if t.ndim != 1:                               # 若监督数据为one-hot编码，则转换为标签索引
      t = np.argmax(t, axis=1)     
    accuracy = np.sum(y == t) / float(x.shape[0]) # 统计预测正确的比例
    return accuracy

  # x: 输入数据, t:监督数据
  def numerical_gradient(self, x, t):
    """
    通过数值微分计算各参数的梯度
    （速度慢，仅用于梯度检查，实际训练应使用 gradient() 方法）
    """

    # 定义一个以权重为变量的损失函数闭包
    loss_W = lambda W: self.loss(x, t)   

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1']) 
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads

  def gradient(self, x, t):
    """
    通过反向传播（误差反向传播法）高效计算各参数的梯度

    参数:
        x : 输入数据
        t : 监督数据
    返回:
        包含各参数梯度的字典 grads
    """
    # ---------- 前向传播 ----------
    self.loss(x, t)                             # 前向传播的同时会存储各层的中间数据，供反向传播使用

    # ---------- 反向传播 ----------
    dout = 1                                    # 从损失函数处开始反向传播，初始导数为1
    dout = self.lastLayer.backward(dout)        # SoftmaxWithLoss层的反向传播

    # 将各中间层反向排列，依次进行反向传播
    layers = list(self.layers.values())         
    layers.reverse()                            # 反向传播需要从输出侧向输入侧依次进行
    for layer in layers:
      dout = layer.backward(dout)               # 每层完成反向传播，并传递梯度

    # 从各层中读取计算好的参数梯度
    grads = {}
    grads['W1'] = self.layers['Affine1'].dW
    grads['b1'] = self.layers['Affine1'].db
    grads['W2'] = self.layers['Affine2'].dW
    grads['b2'] = self.layers['Affine2'].db

    return grads

