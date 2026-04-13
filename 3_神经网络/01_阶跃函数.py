"""
阶跃函数：当输入超过0时，输出1，否则输出0
激活函数从阶跃函数换成其他函数，就可以进入神经网络的世界了

图形
"""
import numpy as np
import matplotlib.pyplot as plt

def step_function(x): # 越阶函数
  return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1) # 在−5.0到5.0的范围内，以0.1为单位，生成NumPy数组（[-5.0, -4.9, ���, 4.9]）
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) #指定y轴的范围
plt.show()
