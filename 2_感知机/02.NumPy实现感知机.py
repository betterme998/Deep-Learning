"""
使用NumPy实现感知机
使用权重和偏置的实现
"""
import numpy as np

def AND(x1, x2): # 与门
  x = np.array([x1, x2]) #输入
  w = np.array([0.5, 0.5]) #权重:是控制输入信号的重要性的参数
  b = -0.7 #偏置：是调整神经元被激活的容易程度 （如b=-0.1,则输入信号的加权总和超过0.1，神经元就会激活）
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

print("与门")
print(AND(0, 0)) # 输出0
print(AND(1, 0)) # 输出0
print(AND(0, 1)) # 输出0
print(AND(1, 1)) # 输出1

# -----------------------

def NAND(x1, x2): # 与非门
  x = np.array([x1, x2])
  w = np.array([-0.5, -0.5]) # 仅权重和偏置与AND不同
  b = 0.7
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1
  
print("与非门")
print(NAND(0, 0)) # 输出1
print(NAND(1, 0)) # 输出1
print(NAND(0, 1)) # 输出1
print(NAND(1, 1)) # 输出0

# -----------------------

def OR(x1, x2): # 或门
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5]) # 仅权重和偏置与AND不同
  b = -0.2
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1
  
print("或门")
print(OR(0, 0)) # 输出0
print(OR(1, 0)) # 输出1
print(OR(0, 1)) # 输出1
print(OR(1, 1)) # 输出1

#-----------------------
#异或门的实现 x1和x2表示输入信号，x1和x2是与非门和或门的输入，而与非门和或门的输出则是与门的输入。  
"""
仅当x1或x2中的一方为1时，才会输出1
   x1,x2|s1,s2|y  
    0, 0| 1, 0|0  
    1, 0| 1, 1|1  
    0, 1| 1, 1|1  
    1, 1| 0, 1|0
"""
def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y

print("异或门")
print(XOR(0,0)) # 输出0
print(XOR(1,0)) # 输出1
print(XOR(0,1)) # 输出1
print(XOR(1,1)) # 输出0

#与门、或门是单层感知机，而异或门是2层感知机。叠加了多层的感知机也称为多层感知机（multi-layered perceptron）。
