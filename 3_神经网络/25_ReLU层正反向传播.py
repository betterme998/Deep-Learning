"""
激活函数ReLU（Rectified Linear Unit）
y = x    # x > 0 
y = 0    # x <= 0
求出y关于x的导数
Ly / Lx = 1    # x > 0 
Ly / Lx = 0    # x <= 0
"""

class Relu:
  """
    ReLU（Rectified Linear Unit，修正线性单元）激活函数层。

    前向传播： out = max(0, x)
    反向传播： dx = dout * (x > 0)
  """
  def __init__(self):
    # 用于存储前向传播时输入 x 的掩码（mask），标记哪些位置的值 <= 0
    self.mask = None

  def forward(self, x): #前向传播。
    self.mask = (x <= 0) # 生成一个布尔掩码，标记输入 x 中所有 <= 0 的位置
    out = x.copy()       # 复制一份 x，避免直接修改原始输入
    out[self.mask] = 0   # 将 <= 0 的元素全部置为 0
    return out