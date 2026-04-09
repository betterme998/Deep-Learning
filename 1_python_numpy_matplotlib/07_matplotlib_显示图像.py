"""
pyplot中还提供了用于显示图像的方法imshow()。另外，可以使用
matplotlib.image模块的imread()方法读入图像。下面我们来看一个例子。
"""
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../image-1.png')
plt.imshow(img)

plt.show()