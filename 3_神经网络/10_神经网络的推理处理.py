"""
神经网络的输入层有784个神经元（图片大小28x28=784）  
输出层有10个神经元。（源于10类别分类（数字0到9）

这个神经网络有2个隐藏层  
第1个隐藏层有50个神经元  
第2个隐藏层有100个神经元  
这个50和100可以设置为任何值。
"""

# coding: utf-8
import sys # 访问命令行参数、系统路径以及Python运行时环境
import pickle  # 用于将数据快速化保存，方便快速加载序列化：
from pathlib import Path # Path 是一个面向对象的文件系统路径处理类
from typing import Any # 类型提示和类型注解的功能

import numpy as np # 科学计算机库，处理数组数据
import numpy.typing as npt # 导入 NumPy 的类型注解工具并起一个简短的别名，主要用于静态类型检查场景，让开发者能清晰地标注数组、标量、数据类型等对象的类型
# 1.使用pathlib 构建路径
# 将父目录添加到系统路径中
# Path(__file__).resolve()当前文件的绝对路径
# parent 获取父级目录

sys.path.append(str(Path(__file__).resolve().parent.parent))
print(str(Path(__file__).resolve().parent.parent))

from dataset.mnist import load_mnist # 从dataset包中导入MNIST数据集加载函数
from common.functions import sigmoid, softmax # 激活函数

def get_data():
  """
  获取经过预处理的MNIST测试数据.
  参数 normalize=True 表示将像素归一化到[0,1] 区间
  flatten=True 表示将 28x28 的图像展平为一维数组（784个元素）；
  one_hot_label=False 表示标签以原始数字形式(0~9) 存储，而非one-hot编码.[0,0,1,0,0,0,0,0,0,0] = 2
  """

