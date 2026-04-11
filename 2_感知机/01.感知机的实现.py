"""
演示与门逻辑电路AND函数,NAND函数，OR函数

注意：这里决定感知机参数的并不是计算机，而是我们人。  
而机器学习的课题就是将这个决定参数值的工作交由计算机自动进行。  
学习是确定合适参数的过程。
而人要做的是思考感知机的构造（模型），并把训练数据交给计算机。

输入信号被送往神经元时 ,会被分别乘以固定的权重(w1x1,w2x2).神经元会计算传过来的信号总和,只有当这个总和超过了某个界限时,才会输出1  
也被称为"神经元激活"  
这里将界限称为"阈值"
"""

def AND(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7 # 权重和阈值
  tmp = x1 * w1 + x2 * w2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1
print("与门")
print(AND(0, 0)) # 输出0
print(AND(1, 0)) # 输出0
print(AND(0, 1)) # 输出0
print(AND(1, 1)) # 输出1

def NAND(x1, x2):
  w1, w2, theta = -0.5, -0.5, -0.7
  tmp = x1 * w1 + x2 * w2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1
print("与非门")
print(NAND(0, 0)) # 输出1
print(NAND(1, 0)) # 输出1
print(NAND(0, 1)) # 输出1
print(NAND(1, 1)) # 输出0

def OR(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.4
  tmp = x1 * w1 + x2 * w2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1
print("或门")
print(OR(0, 0)) # 输出0
print(OR(1, 0)) # 输出1
print(OR(0, 1)) # 输出1
print(OR(1, 1)) # 输出1