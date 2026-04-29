"""
激活函数
"""
import numpy as np

def sigmoid(x):#激活函数
  #避免指数运算溢出
  # 正数区域：使用负指数形式，exp(-x) 很小不溢出
  # 负数区域：使用正指数形式，exp(x) 很小不溢出
  # 1 / (1 + np.exp(-x))和np.exp(x) / (1 + np.exp(x)))这两个写法在数学上完全等价
  return np.where(x >= 0,
                  1 / (1 + np.exp(-x)), #此时exp(-x) <= 1
                  np.exp(x) / (1 + np.exp(x))) #此时exp(x) < 1

# 常用于多分类任务中，把一组实数转换成概率分布（所有输出在0到1之间且和为1）
def softmax(x):#激活函数
  #axis 指定要操作的轴（维度） 
  #-1 在 Python 索引中表示最后一个轴。代表沿着列方向
  # a = np.array([[1, 2, 3],
              # [4, 5, 6]])
  # np.max(a, axis=-1) # 返回 [3, 6]
  # np.max(a, axis=-1, keepdims=True) # 返回 [[3], [6]] 
  #keepdims=True 保留原始数组的维度，而不是默认的降维。 
  x = x - np.max(x, axis=-1, keepdims=True)
  return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims= True)

def cross_entropy_error(y, t): #mini-batch版交叉熵损失函数
  """
  参数：
  y: 预测值，形状为 (N, d) 的数组，其中 N 为批量大小，d 为输出维度,是神经网络的输出
  t: 真实值，形状为 (N, d) 的数组监督数据
  t有两种情况：1. one-hot编码，2. 非one-hot编码
  """
  #同时处理单个数据和批量数据
  if y.ndim == 1: #判断是否为一维数组 #ndim数组维度
    t = t.reshape(1, t.size) #将t转换为1行,t.size列的二维数组,y是神经网络的输出
    y = y.reshape(1, y.size) #将y转换为1行,y.size列的二维数组,t是监督数据
    """
    y的维度为1时，即求单个数据的交叉熵误差时，需要改变数据的形状。
    当输入为mini-batch时，要用batch的个数进行正规化，计算单个数据的平均交叉熵误差。
    """
  batch_size = y.shape[0] # 获取y的行数，即批量大小（求平均值）
  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size