"""
机器学习使用训练数据进行学习  
严格来说，就是针对训练数据计算损失函数的值  
找出使该值尽可能小的参数  
因此计算损失函数时必须将所有的训练数据作为对象  
如果训练数据有100个的话，我们就要把这100个损失函数的总和作为学习的指标

如果数据集有10万张图片的话，计算过程非常耗时  
我们从全部数据中选出一部分，作为全部数据的“近似”  
神经网络的学习也是从训练数据中选出一批数据（称为mini-batch,小批量）  
例： 从60000个训练数据中随机选择100笔，再用这100笔数据进行学习。这种学习方式称为mini-batch学习。

如何从这个训练数据中随机抽取10笔数据呢？

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
"""


