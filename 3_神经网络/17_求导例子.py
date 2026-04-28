"""
数值微分对简单函数进行求导
"""
import numpy as np
import matplotlib.pylab as plt

def function_1(x): # 目标函数-
  return 0.01*x**2 + 0.1*x

def numerical_diff(f, x): # 数值微分
  h = 1e-4 # 0.0001
  return (f(x + h) - f(x - h)) / (2*h) # 中心差分

#绘图
x = np.arange(0.0, 20.0, 0.1) #以0.1为单位，从0到20的数组x
y = function_1(x)
plt.xlabel("x") #x轴的标签
plt.ylabel("f(x)") #y轴的标签
plt.plot(x, y) #绘制图像
plt.show() #显示图像

# 我们来计算一下这个函数在x = 5和x = 10处的导数
print(numerical_diff(function_1, 5)) #0.1999999999990898
print(numerical_diff(function_1, 10)) #0.2999999999986347

"""
f(x) = 0.01*x**2 + 0.1*x 的解析解是df(x)/dx = 0.02*x + 0.1
x=5,x=10处，导数分别为0.2,和0.3 
误差非常小
"""

# 偏导数
# 有多个变量的函数的导数称为偏导数
#f(x0, x1) = x0**2 + x1**2
def function_tmp1(x0): 
  return x0*x0 + 4.0**2.0
def function_tmp2(x1):
  return 3.0**2.0 + x1*x1

# 问题1：求x0 = 3, x1 = 4时，关于x0的偏导数
print(numerical_diff(function_tmp1, 3.0)) # 6.00000000000378

# 问题2：求x0 = 3, x1 = 4时，关于x1的偏导数 。
print(numerical_diff(function_tmp2, 4.0)) # 7.999999999999119
"""
我们定义了一个只有一个变量的函数，并对这个函数进行了求导
问题1中，我们定义了一个固定x1 = 4的新函数，然后对只有变量x0的函数应用了求数值微分的函数

偏导数需要将多个变量中的某一个变量定为目标变量，并将其他变量固定为某个值
"""