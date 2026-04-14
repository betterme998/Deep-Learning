"""
绘制sin函数曲线
这里使用NumPy的arange方法生成了[0, 0.1, 0.2, ..., 5.8, 5.9]的
数据，将其设为x。对x的各个元素，应用NumPy的sin函数np.sin()，将x、
y的数据传给plt.plot方法，然后绘制图形。最后，通过plt.show()显示图形
"""
import numpy as np
import matplotlib.pyplot as plt

#生成数据
x = np.arange(0, 6, 0.1)# 以0.1为单位，生成0到6的数据
y = np.sin(x)
y2 = np.cos(x)

# 绘制图形
plt.plot(x, y, label="sin")
plt.plot(x, y2, linestyle="--",label="cos") #用虚线绘制
plt.xlabel("x") # x轴标签
plt.ylabel("y") # y轴标签
plt.title('sin & cos') # 标题
plt.legend() #用于在 Matplotlib 图表中添加图例，用来区分不同数据系列
plt.show()