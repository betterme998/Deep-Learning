"""
梯度下降
"""
import numpy as np

def _numerical_gradient_1d(f, x):
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x)

  for idx in range(x.size):
    tmp_val = x[idx]
    x[idx] = tmp_val + h
    fxh1 = f(x) # f(x+h)

    x[idx] = tmp_val -h
    fxh2 = f(x) # f(x-h)
    grad[idx] = (fxh1 - fxh2) / (2*h)

    x[idx] = tmp_val
  return grad

def numerical_gradient_2d(f, X):
  if X.ndim == 1:
    return _numerical_gradient_1d(f, X)
  else:
    grad = np.zeros_like(X)

    for idx, x in enumerate(X):
      grad[idx] = _numerical_gradient_1d(f, x)

    return grad
  

def numerical_gradient(f, x):
  h = 1e-4                  #0.0001 # 微小变化量，用于计算数值微分
  grad = np.zeros_like(x)   #初始化梯度数组，形状与 x 相同，元素全为0

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  """
  np.nditer:多维数组迭代器,按内存顺序遍历数组，支持对多个数组同时迭代、广播等高级操作
  flags=['multi_index']:返回多维索引，即每个元素在数组中的位置
  op_flags=['readwrite']:指定操作数组的读写权限
  """
  while not it.finished: #it.finished只读布尔属性判断当前迭代是否已经结束false:未结束
    idx = it.multi_index #it.multi_index返回当前索引
    tmp_val = x[idx]     #保留原始值
    x[idx] = tmp_val + h
    fxh1 = f(x)          #f(x+h)

    x[idx] = tmp_val - h
    fxh2 = f(x)          #f(x-h)
    grad[idx] = (fxh1 - fxh2) / (2*h)#中心差分公式求该分量的偏导数

    x[idx] = tmp_val     #恢复 x[idx] 的原始值，避免影响其他分量的计算
    it.iternext()        #手动将迭代器推进到下一个元素
  return grad
