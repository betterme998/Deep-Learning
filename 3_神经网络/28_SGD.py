"""
随机梯度下降优化器（Stochastic Gradient Descent）。
用数学式可以将SGD写成如下的式  
W <- W - η(偏L/偏W)  
更新的权重参数记为W  
损失函数关于W的梯度记为(偏L/偏W)  
η表示学习率，实际上会取0.01或0.001这些事先决定好的值

SGD是朝着梯度方向只前进一定距离的简单方法

1.SGD的缺点  
梯度的方向并没有指向最小值的方向。  
如果函数的形状非均向（anisotropic），比如呈延伸状，搜索的路径就会非常低效。

这种更新有两个典型问题：  
锯齿震荡：当损失函数的“地形”是狭长山谷（病态曲率）时，梯度方向会频繁拐弯，导致参数在垂直于谷底的方向上来回震荡，前进缓慢。

停滞在平坦区/鞍点：梯度过小甚至为零时，参数几乎不再更新，训练被困住。
"""

class SGD:
  # 最简单的参数更新方式：param = param - learning_rate * gradient
  def __init__(self, lr=0.01):
    """
    初始化优化器。
    
    参数：
        lr: float, 学习率（learning rate），控制参数更新的步长。
    """
    self.lr = lr

  def update(self, params, grads):
    """
    根据梯度更新参数。
    
    参数：
        params: dict, 模型参数字典，键为参数名，值为参数值（如numpy数组）。
        grads:  dict, 梯度字典，键与params相同，值为对应参数的梯度。
    """
    for key in params.keys():
      # 核心更新公式：参数 -= 学习率 × 梯度
      params[key] -= self.lr * grads[key]