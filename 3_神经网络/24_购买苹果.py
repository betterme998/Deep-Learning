"""
实现购买苹果的例子。这里，我们把要实现的计算图的
乘法节点称为“乘法层”（MulLayer），
加法节点称为“加法层”（AddLayer）
"""
class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None

  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x * y

    return out
  
  def backward(self, dout):
    dx = dout * self.y
    dy = dout * self.x

    return dx, dy
  
class AddLayer:
  def __init__(self):
    pass

  def forward(self, x, y):
    out = x + y

    return out
  
  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1

    return dx, dy
  
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1  # 税率

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_orange_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num) #(1)
orange_price = mul_orange_layer.forward(orange, orange_num) #(2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price) #(3)
price = mul_tax_layer.forward(all_price, tax) #(4)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice) #(4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price) #(3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price) #(2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price) #(1)

print("price:", int(price)) # price: 220
print("dApple:", dapple) # dApple: 2.2
print("dApple_num:", int(dapple_num)) # dApple_num: 110
print("dTax:", dtax) # dTax: 200

print(price) # 715
print(dapple_num, dapple, dorange, dorange_num, dtax) # 110 2.2 3.3 165 650
