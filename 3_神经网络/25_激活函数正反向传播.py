"""

"""
import numpy as np
from common.functions import *


class Relu:
  """
    激活函数ReLU（Rectified Linear Unit）
    y = x    # x > 0 
    y = 0    # x <= 0
    求出y关于x的导数
    Ly / Lx = 1    # x > 0 
    Ly / Lx = 0    # x <= 0

    ReLU（Rectified Linear Unit，修正线性单元）激活函数层。

    前向传播： out = max(0, x)
    反向传播： dx = dout * (x > 0)
  """
  def __init__(self):
    # 用于存储前向传播时输入 x 的掩码（mask），标记哪些位置的值 <= 0
    self.mask = None

  def forward(self, x): #前向传播。
    self.mask = (x <= 0) # 生成一个布尔掩码，标记输入 x 中所有 <= 0 的位置
    out = x.copy()       # 复制一份 x，避免直接修改原始输入
    out[self.mask] = 0   # 布尔索引,所有在 self.mask 里为 True 的对应位置, 将 <= 0 的元素全部置为 0
    return out
  
  def backward(self, dout):
    """
      反向传播，计算损失关于输入 x 的梯度。

      参数：
        dout : numpy.ndarray
            损失关于该层输出的梯度（上游梯度）。

      返回：
        dx : numpy.ndarray
            损失关于该层输入的梯度（下游梯度）。
    """
     # 前向传播时 <= 0 的位置，梯度为 0（因为输出为常数 0，不传递梯度）
    dout[self.mask] = 0#布尔索引 将那些 ≤ 0 位置的梯度清零，然后 dx = dout 把修改后的梯度作为下游梯度返回。

     # 梯度直接传递（因为 out = x 的部分导数为 1）
    dx = dout

    return dx
  

class Sigmoid:
  def __init__(self):
    # 保存前向传播的输出，供反向传播时使用
    self.out = None

  def forward(self, x):
    """
    前向传播：计算 sigmoid 激活值。
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    out = sigmoid(x) # sigmoid 为外部定义的函数，如 1/(1+np.exp(-x))
    self.out = out # 缓存输出，用于反向传播求导
    return out
  
  def backward(self, dout):
    """
    反向传播：根据上游传来的梯度 dout（损失关于该层输出的梯度），
    计算损失关于输入 x 的梯度 dx 并返回。
    
    sigmoid 函数的导数：sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    因此 dx = dout * self.out * (1 - self.out)
    """
    dx = dout * (1.0 - self.out) * self.out
    return dx
  

class Affine:
  def __init__(self, W, b):
    # 权重矩阵W，形状一般为（输入特征数，输出特征数）
    self.W = W
    # 偏置向量 b，形状为 (输出特征数,) 或 (1, 输出特征数)
    self.b = b

    # 保存前向传播时的输入，供反向传播计算梯度使用
    self.x = None
    # 保存输入数据的原始形状，用于将梯度恢复为多维形状（比如图像张量）
    self.original_x_shape = None

    # 权重和偏置参数的梯度，反向传播时计算并存储
    self.dW = None
    self.db = None

  def forward(self, x):
    """
    前向传播：计算仿射变换（全连接） out = x·W + b
    支持输入为多维张量，会将除了 batch 维度外的维度展平。
    """
    # 记录原始形状，便于反向传播时将 dx 恢复成相同形状
    self.original_x_shape = x.shape
    # 将输入展平为 (batch_size, 特征数)
    # 例如，如果输入形状为 (2, 3, 4)，展平后为 (2, 12)
    x = x.reshape(x.shape[0], -1)
    self.x = x

    # 线性变换：矩阵乘法 + 偏置
    out = np.dot(self.x, self.W) + self.b
    return out
  
  def backward(self, dout):
    """
    反向传播：根据上游传来的梯度 dout（损失对输出的梯度），
    计算：
    - dx：损失对输入 x 的梯度
    - 同时累加（或直接计算）权重梯度 self.dW 和偏置梯度 self.db
    """
    # dx = dout · W^T，形状 (batch_size, 输入特征数)
    dx = np.dot(dout, self.W.T)
    # dW = x^T · dout，形状与 W 相同
    self.dW = np.dot(self.x.T, dout)
    # db = 对 batch 维度求和，形状与 b 相同
    self.db = np.sum(dout, axis=0)

    # 将 dx 的形状还原为 forward 时输入 x 的原始形状（适应卷积、图像等张量输入）
    # forward中多维的输入（例如图像、特征图）强行拉成了一个二维矩阵
    """
     dx = np.dot(dout, self.W.T)
    这个 dx 是损失对展平后输入的梯度，形状为二维。
    但是，前面的层可能期望收到形状为 (100, 1, 28, 28) 的梯度
    （例如卷积层、池化层，或者单纯是为了保持张量结构）。
    如果直接返回 (100, 784)，就会出现形状不匹配的错误，反向传播无法继续。
    """
    # * 是 Python 的解包操作符，它会把元组中的元素拆开，作为独立的参数传递给函数。
    dx = dx.reshape(*self.original_x_shape)
    return dx