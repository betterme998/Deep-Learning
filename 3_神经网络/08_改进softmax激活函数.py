"""
实现softmax函数时的注意事项  
溢出问题:指数运算
exp(x)的输入x的范围是[-709.782, 709.782]

改进的softmax函数 
yk = exp(ak ) / sum(exp(ai))
yk = Cexp(ak ) / Csum(exp(ai)) 
yk = exp(ak + c') / sum(exp(ai + c'))  
分子和分母上都乘上C这个任意的常数  
把这个C移动到指数函数（exp）

加上（或者减去）某个常数并不会改变运算的结果  
C'可以使用任何值，但是为了防止溢出，一般会使用输入信号中的最大值
"""
import numpy as np
a = np.array([1010, 1000, 990])

def softmax(a):
  c = np.max(a) # 1010
  exp_a = np.exp(a-c) # 溢出对策
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y 
print(softmax(a)) # [9.99954600e-01 4.53978686e-05 2.06106005e-09]