"""
整合三层神经网络代码
"""
import numpy as np
#sigmoid函数
def sigmoid(x): # 参数为NumPy数组
  return 1/(1+np.exp(-x))
def identity_function(x): #激活函数： 恒等函数
  return x

def init_network(): #权重和偏置初始化
  #字典变量network中保存了每一层所需的参数（权重和偏置）
  network = {}
  #第一层权重和偏置
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  #第二层权重和偏置
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) 
  network['b2'] = np.array([0.1, 0.2])
  #第三层权重和偏置
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])
  return network

def forward(network, x):# 向前传播
  # 封装了将输入信号转换为输出信号的处理过程。
  W1, W2, W3 = network['W1'], network['W2'], network['W3'] 
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1 # 加权求和
  Z1 = sigmoid(a1) # 激活函数
  a2 = np.dot(Z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = identity_function(a3) 
  return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) #[0.31682708 0.69627909]

#输出层所用的激活函数，要根据求解问题的性质决定
# 回归问题可以使用恒等函数
# 二元分类问题可以使用sigmoid函数
#多元分类问题可以使用softmax函数