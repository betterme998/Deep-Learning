"""
我们把f(x0+x1)= x0**2 + x1**2 的梯度画在图上。不过，这里我们画的是元素值为负梯度的向量
"""
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def _numerical_gradient_no_batch(f, x):
  """
  对单个点x计算函数f的数值梯度（中心差分法）

  参数：
  f: 待函数，接受一个与 x 相同形状的数组，返回一个标量
  x:  一维 numpy 数组，当前点（要求梯度的点）

  返回:
        grad : 与 x 形状相同的梯度向量
  梯度向量
  """
  h = 1e-4 # 微小变化量，用于计算数值微分
  grad = np.zeros_like(x) # 初始化梯度数组，形状与 x 相同, 元素全为0
   # 对 x 的每一个分量分别计算偏导数
  for idx in range(x.size):
      tmp_val = x[idx]  # 保存原始值

      # 计算 f(x + h)
      x[idx] = float(tmp_val) + h
      fxh1 = f(x)

      # 计算 f(x - h)
      x[idx] = tmp_val - h
      fxh2 = f(x)

      # 中心差分公式求该分量的偏导数
      grad[idx] = (fxh1 - fxh2) / (2 * h)

      # 恢复 x[idx] 的原始值，避免影响其他分量的计算
      x[idx] = tmp_val
  return grad

def numerical_gradient(f, X):
   """
   计算函数 f 在点集 X 上的数值梯度，支持批量输入
   参数:
    f : 函数
    X : 一维数组（单个点）或二维数组（多个点的集合，每行一个点）

  返回:
        grad : 与 X 形状相同的梯度数组
   """
   if X.ndim == 1: 
      # 如果 X 是一维数组，说明是单个点，直接调用单点梯度计算
      return _numerical_gradient_no_batch(f, X)
   else:
      # 如果是二维数组，逐行（逐个点）计算梯度
      grad = np.zeros_like(X)
      for idx, x in enumerate(X):
         grad[idx] = _numerical_gradient_no_batch(f, x)
      return grad
   
def function_2(x):
  """
  示例函数：f(x) = sum(x_i^2) ，即所有分量的平方和

  参数:
      x : 一维或二维数组，二维时每行是一个点
  返回:
      函数值，标量或一维数组（批量时返回每个点的函数值）
  """
  if x.ndim == 1:
   return np.sum(x**2)
  else:
     # 当输入为二维时，按行求和，axis=1 表示保留每一行
     return np.sum(x**2, axis=1)
  
def tangent_line(f, x):
   """
   求函数 f 在点 x 处的切线的函数表示（仅用于单变量或可广播的情况）
   注意：本函数在下面的主程序中并未使用
   参数:
        f : 函数
        x : 求切线的点

   返回:
        一个 lambda 函数，输入 t 返回切线在 t 处的值
   """
   d = numerical_gradient(f, x)  # 计算该点的梯度（斜率）
   print(d)
   y = f(x) - d * x  # 切线方程的截距项（向量计算，利用了广播）
   return lambda t: d * t + y  # 返回切线函数

if __name__ == '__main__':
  # 生成二维网格坐标
  x0 = np.arange(-2, 2.5, 0.25)  # x0 轴方向的坐标点
  x1 = np.arange(-2, 2.5, 0.25)  # x1 轴方向的坐标点
  X, Y = np.meshgrid(x0, x1)     # 生成网格矩阵 X, Y 从坐标向量生成坐标矩阵的函数，常用于二维或三维网格的创建，

  # 将网格矩阵展平为一维数组，方便批量计算
  X = X.flatten() # 将多维数组降为一维数组
  Y = Y.flatten()

  # 将所有网格点组合成二维点的集合（每一行是一个点 [x0, x1]）
  # 计算所有点的梯度，然后转置回来，使 grad[0] 是所有点的 x0 方向梯度，grad[1] 是 x1 方向梯度
  grad = numerical_gradient(function_2, np.array([X, Y]).T).T

  # 绘制梯度向量场（使用 quiver 画箭头）
  plt.figure()
  # -grad[0], -grad[1] 表示负梯度方向，即函数值下降最快的方向
  plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
  plt.xlim([-2, 2])
  plt.ylim([-2, 2])
  plt.xlabel('x0')
  plt.ylabel('x1')
  plt.grid()
  plt.draw()
  plt.show()
         