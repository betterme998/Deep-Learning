"""
我们以一个简单的神经网络为例，来实现求梯度的代码
simpleNet类只有一个实例变量，即形状为2×3的权重参数
它有两个方法，一个是用于预测的predict(x)，
另一个是用于求损失函数值的loss(x,t)。这里参数x接收输入数据，t接收正确解标签。

numerical_gradient(f, x) 的参数f是函数，x是传给函数f的参数。
因此，这里参数x取net.W，并定义一个计算损失函数的新函数f，
然后把这个新定义的函数传递给numerical_gradient(f, x)。
"""
import sys
import os
import numpy as np
from numpy.random import default_rng
# 将父目录添加到 sys.path，确保可以导入 common 模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # os.path.join()函数用于路径拼接文件路径，可以传入多个路径
print(os.path.join(os.path.dirname(__file__), '..')) # '..' 表示上一级目录，得到父目录的路径。

from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class SimpleNet:
  """不含隐藏层的简单神经网络"""
  def __init__(self, rng: np.random.Generator | None = None) -> None: # 可选地传入一个 NumPy 随机数生成器对象（例如通过 np.random.default_rng() 创建）
    if rng is None:
      rng = default_rng()
    """
    rng.standard_normal((2, 3))：调用随机数生成器 rng 的 standard_normal 方法，生成形状为 (2, 3) 的数组。
    这里 rng 通常是 numpy.random.Generator 的实例（例如通过 np.random.default_rng() 创建），
    standard_normal 会生成独立同分布的标准正态随机数。
    self.W 的结果类似于：
    array([[ 0.123, -0.456,  0.789],
       [-0.234,  1.567, -0.678]])
    """
    self.W: np.ndarray = rng.standard_normal((2, 3)) # 权重矩阵

  def predict(self, x: np.ndarray) -> np.ndarray:
    """前向传播：线性输出"""
    return np.dot(x, self.W)
  
  def loss(self, x: np.ndarray, t: np.ndarray) -> float:
    """交叉熵损失"""
    z = self.predict(x) # 
    y = softmax(z) #
    return cross_entropy_error(y, t)
  
# 输入数据与标签
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1]) #  # one-hot，正确类别为索引 2

# 创建网络
net = SimpleNet()
print(net.W) #权重参数
"""
[[ 0.4312431   0.04586049  0.45933726]
 [-0.75540415  1.16471129  2.05884802]]
"""

p = net.predict(x)
print(p) #[-0.42111787  1.07575645  2.12856558]
print(np.argmax(p)) #最大值的索引 2

print(net.loss(x, t))#交叉熵损失 7.2190263601840705

# 损失函数关于权重的梯度（数值微分）
f = lambda w: net.loss(x, t) # 一个匿名函数（lambda），参数为 w,返回当前网络在输入 x 和正确标签 t 下的交叉熵损失。
dW = numerical_gradient(f, net.W)#参数：f损失函数， net.W（当前权重矩阵）

print(dW)
"""
梯度
[[-0.10169235  0.02934227  0.07235007]
 [-0.15253852  0.04401341  0.10852511]]

 w11大约为-0.1，这表示如果将w11增加h，那么损失函数的值会减少0.1h
 w23大约为0.1，这表示如果将w23增加h，那么损失函数的值会增加0.1h

"""

