"""
有多个变量的函数的导数称为偏导数
过个变量一起求导，生成的向量称为梯度
我们希望一起计算x0和x1的偏导数。比如，我们来考虑求x0 = 3, x1 = 4时(x0, x1)的偏导数
这样的由全部变量的偏导数汇总而成的向量称为梯度
"""
import numpy as np

def function_2(x): # 目标函数-(就是要求这个函数的导数)
  return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
  """
   使用中心差分法计算函数 f 在点 x 处的数值梯度。

   参数:
        f : 函数，接受一个与 x 同形状的 numpy 数组作为输入
        x : numpy 数组，待求梯度的点
   返回:
        grad : numpy 数组，与 x 形状相同，f 在 x 处的梯度近似值
  """
  h = 1e-4 #0.0001 # 微小增量，用于近似导数的差分步长
  grad = np.zeros_like(x) #生成和x形状相同的数组,并且所有元素都用 0 填充。

  for idx in range(x.size): # # 对 x 的每一个分量逐个计算偏导数
    tmp_val = x[idx] # # 1. 保存当前分量的原始值，后续要用来恢复
    # 2. 计算 f(x + h) 
    #    将第 idx 个分量替换为 (原值 + h)
    x[idx] = tmp_val + h
    fxh1 = f(x) # # 此时 x 变成了 (x1, ..., xi+h, ...)

    # 3. 计算 f(x - h)
    #    将第 idx 个分量替换为 (原值 - h)
    #    注意：这里使用的是之前保存的原始值 tmp_val，而不是上面已经变过的值
    x[idx] = tmp_val -h
    fxh2 = f(x) # 此时 x 变成了 (x1, ..., xi-h, ...)

    grad[idx] = (fxh1 - fxh2) / (2*h) # # 4. 用中心差分公式计算该分量上的偏导数
    x[idx] = tmp_val # # 5. 恢复该分量为原始值，保证下一轮循环开始时 x 是干净的
  return grad

# 这里我们求点(3, 4)、(0, 2)、(3, 0)处的梯度
print(numerical_gradient(function_2,np.array([3.0,4.0])))#[6. 8.] 
print(numerical_gradient(function_2,np.array([0.0,2.0])))#[0. 4.]
print(numerical_gradient(function_2,np.array([3.0,0.0])))#[6. 0.]

# 实际上，虽然求到的值是[6.0000000000037801, 7.9999999999991189]，
# 但实际输出的是[6., 8.]。这是因为在输出NumPy数组时，数值会被改成“易读”的形式。
# 负梯度方向是梯度法中变量的更新方向