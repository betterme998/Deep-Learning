"""
训练图像的第一张就会显示出来
"""
# coding: utf-8
# 导入系统与操作系统相关的模块，用于处理文件路径
import sys, os
from pathlib import Path
import numpy as np # 导入NumPy，用于高效的数组和矩阵运算
# 将父目录添加到系统路径中，以便能够导入父目录下的模块（如dataset.mnist）
sys.path.append(str(Path(__file__).parent.parent))
print(str(Path(__file__).parent))

from dataset.mnist import load_mnist # 从dataset包中导入MNIST数据集加载函数
import matplotlib.pyplot as plt



# 定义一个显示图像的函数
def img_show(img):
  """
  将NumPy数组转换为PIL图像对象并显示
  参数img：二维NumPy数组，表示灰度图像，像素值范围通常为0~255
  """

  #Image.fromarray要求数据类型为unit8，这里进行显示转换
  # np.uint8 是 NumPy 提供的一种 无符号 8 位整数类型（Unsigned 8-bit Integer），
  # 取值范围为 0~255，占用 1 个字节 内存。它常用于 图像处理、嵌入式数据存储 等需要节省内存的场景。
  # pil_img = Image.fromarray(np.uint8(img))
  # pil_img.show() # 调用系统默认图像查看器显示图片

  """使用 Matplotlib 显示灰度图像"""
  plt.imshow(img, cmap='gray')
  plt.axis('off')
  plt.show()

# 加载MNIST数据集
# flatten=True: 将28x28的图像展平为一维数组（长度784），方便作为全连接网络的输入
# normalize=False: 不进行归一化，像素值保持原始的0~255整数，便于直接显示
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

#取出第一张训练图像及其对应的标签
img = x_train[0] # 图像数据，形状为(784,)的一维数组
label = t_train[0]  # 标签，整数0~9
print(label) # # 打印标签，预期输出：5（MNIST第一张图片是数字5）
print(img.shape) # # 打印图像数据的形状，输出：(784,) 确认是展平的一维数组

# 将展平的图像数据重新变形为28x28的二维数组，恢复原始图像尺寸
img = img.reshape(28, 28)
print(img.shape) # # 打印变形后的形状，输出：(28, 28)

#调用自定义函数显示图像
img_show(img)