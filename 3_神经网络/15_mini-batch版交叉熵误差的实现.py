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
  return -np.sum(np.log([np.arange(batch_size, t)] + 1e-7)) / batch_size


