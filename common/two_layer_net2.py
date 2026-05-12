"""
使用了激活函数层来计算
2层神经网络（隐藏层为1层的网络）为对象，使用MNIST数据集进行学习。
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))# 将父目录加入路径，以便导入common文件夹中的模块
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict # OrderedDict 是一种能记住键插入顺序的字典。

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # 初始化权重
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

    # 生成层
    self.layers = OrderedDict()
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
    self.layers['Relu1'] = Relu()
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x):
    for layer in self.layers.values():
      x = layer.forward(x)
    
    return x
  
  # x: 输入数据, t:监督数据
  def loss(self, x, t):
    y = self.predict(x)
    return self.lastLayer.forward(y, t)
  
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    if t.ndim != 1 : t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  # x: 输入数据, t:监督数据
  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)   

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1']) 
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

