import numpy as np
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
"""
[[51 55]
 [14 19]
 [ 0  4]]
"""

print(X[0])
"""
[51 55]
"""

print(X[0][1]) # 55

for row in X:
  print(row)
"""
[51 55]
[14 19]
[0 4]
"""

# NumPy还可以使用数组访问各个元素。
X = X.flatten() #将X转换为一维数组
print(X) # [51 55 14 19  0  4]

print(X[np.array([0, 2, 4])]) # [51 14  0]

# 从X中抽出大于15的元素
print(X > 15) # [ True  True False  True False False]
print(X[X > 15]) # [51 55 19]