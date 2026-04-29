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
      output_size:
    """