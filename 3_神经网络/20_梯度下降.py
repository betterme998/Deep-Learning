"""
神经网络必须在学习时找到最优参数（权重和偏置）。
这里所说的最优参数是指损失函数取最小值时的参数.
梯度法: 通过计算损失函数对每个参数的偏导数，并将这些偏导数乘以负号，然后再减去学习率，最后得到新的参数

在梯度法中，函数的取值从当前位置沿着梯度方向前进一定距离，
然后在新的地方重新求梯度，再沿着新梯度方向前进，如此反复，不断地沿梯度方向前进。
像这样，通过不断地沿梯度方向前进，逐渐减小函数值的过程就是梯度法（gradient method）

寻找最小值的梯度法称为梯度下降法（gradient descent method），
寻找最大值的梯度法称为梯度上升法（gradient ascent method）。
神经网络（深度学习）中，梯度法主要是指梯度下降法

数学式来表示梯度法:
x0 = x0 - η * ∂f(x0)/∂x0
x1 = x1 - η * ∂f(x1)/∂x1

η表示更新量,称为学习率（learning rate）
学习率:决定在一次学习中，应该学习多少，以及在多大程度上更新参数(如：0.01或0.001)
"""
import numpy as np
# 求导
def _numerical_gradient_no_batch(f, x):
  """
  对单个点x计算函数f的数值梯度（中心差分法）

  参数：
  f: 待函数，接收一个与 x 相同形状的数组，返回一个标量
  x: 一维 numpy 数组，当前点（要求梯度的点）

  返回：
  grad ： 与 x 形状相同的梯度向量
  """

  h = 1e-4 # 微小变化量，用于计算数值微分
  grad = np.zeros_like(x)             #初始化梯度数组，形状与 x 相同，元素全为0
  for idx in range(x.size):           #对 x 的每个分量分别计算偏导数
    tmp_val = x[idx]                  #保留原始值
    x[idx] = float(tmp_val) + h       #计算f(x + h)
    fxh1 = f(x)

    x[idx] = tmp_val - h              #计算f(x - h)
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h) #中心差分公式求该分量的偏导数
    x[idx] = tmp_val                  #恢复 x[idx] 的原始值，避免影响其他分量的计算
  return grad

def numerical_gradient(f, X):
  """
  计算函数 f 在点集 X 上的数值梯度，支持批量输入
  参数：
  f ：函数
  X ：一维数组（单个点）或二维数组（多个点的集合，每行一个点）

  返回：
  grad ：与 X 形状相同的梯度数组
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




# 梯度下降法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
  """
  参数：
  f:要进行最优化的函数
  nit_x:是初始值
  lr是学习率learning rate
  step_num是梯度法的重复次数
  numerical_gradient(f,x)会求函数的梯度，
  用该梯度乘以学习率得到的值进行更新操作，
  由step_num指定重复的次数。

  使用这个函数可以求函数的极小值
  """
  x = init_x
  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
  return x

# 例子：请用梯度法求f(x0 +x1) = x0**2 + x1**2的最小值
def function_2(x):
  return x[0]**2 + x[1]**2
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))# [-6.11110793e-10  8.14814391e-10]

"""
# 像学习率这样的参数称为超参数
这是一种和神经网络的参数（权重和偏置）性质不同的参数。
相对于神经网络的权重参数是通过训练数据和学习算法自动获得的，
学习率这样的超参数则是人工设定的。
一般来说，超参数需要尝试多个值，以便找到一种可以使学习顺利进行的设定。
"""