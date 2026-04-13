"""
sigmoid函数的实现
sigmoid函数和阶跃函数的比较  
.sigmoid函数是一条平滑的曲线，输出随着输入发生连续性的变化。  
.阶跃函数以0为界，输出发生急剧性的变化

"""

import numpy as np
import matplotlib.pyplot as plt

#sigmoid函数
def sigmoid(x): # 参数为NumPy数组
  return 1/(1+np.exp(-x))
#标量和NumPy数组的各个元素进行了运算，运算结果以NumPy数组的形式被输出

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)#指定y轴的范围
plt.show()

