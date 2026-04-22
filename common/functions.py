"""
激活函数
"""
import numpy as np

def sigmoid(x):
  #避免指数运算溢出
  # 正数区域：使用负指数形式，exp(-x) 很小不溢出
  # 负数区域：使用正指数形式，exp(x) 很小不溢出
  # 1 / (1 + np.exp(-x))和np.exp(x) / (1 + np.exp(x)))这两个写法在数学上完全等价
  return np.where(x >= 0,
                  1 / (1 + np.exp(-x)), #此时exp(-x) <= 1
                  np.exp(x) / (1 + np.exp(x))) #此时exp(x) < 1

# 常用于多分类任务中，把一组实数转换成概率分布（所有输出在0到1之间且和为1）
def softmax(x):
  #axis 指定要操作的轴（维度） 
  #-1 在 Python 索引中表示最后一个轴。代表沿着列方向
  # a = np.array([[1, 2, 3],
              # [4, 5, 6]])
  # np.max(a, axis=-1) # 返回 [3, 6]
  # np.max(a, axis=-1, keepdims=True) # 返回 [[3], [6]] 
  #keepdims=True 保留原始数组的维度，而不是默认的降维。 
  x = x - np.max(x, axis=-1, keepdims=True)
  return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims= True)