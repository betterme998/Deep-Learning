"""
softmax函数：将多维数组的元素转换成总和为1的概率分布  
yk = exp(ak) / sum(exp(a1), exp(a2), ..., exp(ak)):计算第k个神经元的输出yk (分母加起来为1，求概率)  
exp(x)是表示ex的指数函数（e是纳皮尔常数2.7182 . . .）  
softmax函数的分子是输入信号ak的指数函数，分母是所有输入信号的指数函数的和。  
softmax函数:输出层的各个神经元都受到所有输入信号的影响.

演示softmax函数
"""
import numpy as np

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a) # 指数函数
print(exp_a) # [ 1.34985881 18.17414537 54.59815003]

sum_exp_a = np.sum(exp_a) #指数函数的和
print(sum_exp_a) # 74.1221542101633

y = exp_a / sum_exp_a
print(y) # [0.01821127 0.24519181 0.73659691]

def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y

# 实现softmax函数时的注意事项
