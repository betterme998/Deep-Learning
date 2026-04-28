"""
mini-batch版交叉熵误差的实现
Loss =- (1/N) * -∑(t * log(ŷ_i))

同时处理单个数据和批量数据
"""
import numpy as np

#输入one-hot
def cross_entropy_error(y, t):
  # 同时处理单个数据和批量数据
  if y.ndim == 1: #判断是否为一维数组 #ndim数组维度
    t = t.reshape(1, t.size) # 将t转换为1行,t.size列的二维数组,y是神经网络的输出
    y = y.reshape(1, y.size) # 将y转换为1行,y.size列的二维数组,t是监督数据
    """
    y的维度为1时，即求单个数据的交叉熵误差时，需要改变数据的形状。
    当输入为mini-batch时，要用batch的个数进行正规化，计算单个数据的平均交叉熵误差。
    """
  
  batch_size = y.shape[0] # 获取y的行数,即batch_size （求平均值）
  return -np.sum(t * np.log(y + 1e-7)) / batch_size
  #y是神经网络的输出
  #t是监督数据
  #y的维度为1时，即求单个数据的交叉熵误差时，需要改变数据的形状。
  #当输入为mini-batch时，要用batch的个数进行正规化，计算单个数据的平均交叉熵误差。

# 监督数据是标签形式（非one-hot表示，而是像“2”“7”这样的标签）时
def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  """
  y[np.arange(batch_size, t)]到底取了什么？
  # y是神经网络的输出
  假设我们批有100个样本，共有10个类别，那么y的形状是(100, 10)
  样本0: [0.1, 0.2,... 0.6, 0.1]   ← 类别0~3的概率
  样本1: [0.3, 0.4,... 0.1, 0.2]
  ...
  样本100: [0.1, 0.1, ...0.1, 0.7]

  t是真实标签的类别索引，形状(100,)
  t = [2, 7, 3, ..., 5]
  意思是：t[i]表示样本i的真实概率最高的类别

  那么y[np.arange(batch_size, t)]等价于
  y[0, 2]
  y[1, 7]
  ...
  y[100, 5] 前一项是样本索引，后一项是真实标签的类别索引(就是最高概率的那个)
  取出来的是一个数组[0.6,0.4,...,0.7]

  这里的t只是为了取出y中对应真实标签的概率
  np
  """
  return -np.sum(np.log(y[np.arange(batch_size, t)] + 1e-7)) / batch_size


