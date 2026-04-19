# coding: utf-8
# 导入网络请求库，用于下载MNIST数据集文件
try:
  import urllib.request
except ImportError:
  #raise主动抛出异常
  raise ImportError('You should use Python 3.x')# 仅支持Python 3.x
import os.path # 专门用于处理文件路径（例如拼接路径、判断文件是否存在、获取文件名/目录名等）。
import gzip    # 用于解压.gz文件
import pickle  # 用于将数据快速化保存，方便快速加载序列化：
#pickle功能
#序列化：将 Python 对象（如列表、字典、自定义类实例等）转换为字节流，以便存储到文件或通过网络传输。
#反序列化：从字节流中还原出原来的 Python 对象
import os #操作文件和目录,获取系统信息,执行系统命令,管理进程,处理路径
import numpy as np #科学计算机库，处理数组数据

# MNIST数据集的官方下载地址（已替换为镜像站，国内访问更快）
# url_base = 'http://yann.lecun.com/exdb/mnist/'
url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # 镜像站点

# 定义四个关键文件的名称（压缩包格式）
key_file = {
  'train_img': 'train-images-idx3-ubyte.gz',   # 训练图像
  'train_label': 'train-labels-idx1-ubyte.gz', # 训练标签
  'test_img': 't10k-images-idx3-ubyte.gz',     # 测试图像
  'test_label': 't10k-labels-idx1-ubyte.gz'    # 测试标签
}

# 获取当前脚步文件所在的目录，作为数据集存储路径
dataset_dir = os.path.dirname(os.path.abspath(__file__))
# 最终保存的pickle文件名，包含所有处理后的数据
save_file = dataset_dir + "/mnist.pkl"

# MNIST数据集基本信息
train_num = 60000       # 训练样本总数
test_num = 10000        # 测试样本总数
img_dim = (1, 28, 28)   # 图像维度：通道数1（灰度），高28，宽28
img_size = 784          # 28*28 = 784，将二维图像展平为一维向量的长度

def _download(file_name):
  """
  下载单个文件到本地目录（如果文件已存在则跳过）
  参数：
      file_name: 要下载的文件名（如 'train-images-idx3-ubyte.gz'）
  """
  file_path = dataset_dir + "/" + file_name

  # 若文件已存在，无需重复下载
  if os.path.exists(file_name): # 文件路径存在性检查
    return
  
  print("Downloading" + file_name + "...")
  # 添加User-Agent头，模拟浏览器访问，避免某些服务器拒绝爬虫
  headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"}
  # 构造请求对象
  # urllib.request.Request 就是 Python 自带的一个请求构造器，让你能把 URL、Headers、数据打包成一个标准请求，再发给服务器。它是爬虫伪装成浏览器的第一步。
  request = urllib.request.Request(url_base + file_name, headers=headers)
  # 发送请求并读取响应内容（二进制数据）
  # 返回的是一个HTTPResponse 对象
  # 此时，数据还在“水管”里源源不断地流过来，并没有一次性全部下载到你的内存里
  # .read() 就是拧开水龙头
  # HTTPResponse 对象有一个方法叫 .read()，它的作用就是从数据通道里把服务器发来的字节全部读取到内存中，并返回给你。
  response = urllib.request.urlopen(request).read()

  # 将下载内容写入本地文件（'wb'模式表示二进制写入）
  with open(file_path, mode='wb') as f:
    f.write(request)
  print("Done")


def download_mnist():
  """"下载所有必须的MNIST数据文件"""
  for v in key_file.values():
    _download(v)

def _load_label(file_name):
  """
  从.gz压缩文件中加载标签数据，并转换为NumPy数组
  参数:
      file_name: 标签文件名（如 'train-labels-idx1-ubyte.gz'）
  返回:
      labels: 一维NumPy数组，元素为0~9的整数标签
  """
  file_path = dataset_dir + "/" + file_name

  print("Converting" + file_name + "to NumPy Array ...")
  # 以二进制读模式打开gzip压缩文件
  with gzip.open(file_path, 'rb') as f:
    # 标签文件格式：前8个字节是文件头信息（魔数，标签数量等，之后才是真正的标签数据）
    # np.frombuffer 用于将一段字节缓冲区直接解释为一维数组，不复制数据（只创建视图）。
    # 参数一：缓冲区，这里是 f.read() 返回的 bytes 对象。
    # 参数二：dtype=np.uint8 表示每个标签占用1个字节，被解释为一个 0–255 的无符号整数（8 位）
    # 参数三：offset=8 表示跳过前8个字节，从第9个字节开始读取
    labels = np.frombuffer(f.read(), np.uint8, offset=8)
  print("Done")
  return labels


def _load_img(file_name):
  """
  从.gz压缩文件中加载图像数据，并转化为NumPy数组
  参数：
    file_name: 图像文件名（如 'train-images-idx3-ubyte.gz'）
  返回：
    data: 二维NumPy数组，形状为（样本数, 784）,每个元素是0~255的像素值
  """ 
  file_path = dataset_dir + "/" + file_name

  print("converting" + file_path + "to NumPy Array ...")
  with gzip.open(file_path, 'rb') as f:
    # 图像文件格式：前16个字节是文件头信息（魔数,图像数量，行数，列数等）
    # offset=16 表示跳过这16个字节，直接读取像素数据
    data = np.frombuffer(f.read(), np.uint8, offset=16)
  # 将一维像素数组重塑为二维：（样本数，784）
  # -1 表示自动计算样本数量， img_size = 784  
  data = data.reshape(-1, img_size)
  print("Done")
  return data

def _convert_numpy():
  """
  将下载的原始文件转换为NumPy数组，并组织成字典格式
  返回：
    dataset:包含四个键的字典：
      'train_img'   : 训练图像数组
      'train_label' : 训练标签数组
      'test_img'    : 测试图像数组
      'test_label'  : 测试标签数组
  """
  dataset = {}
  dataset['train_img'] = _load_img(key_file['train_img'])
  dataset['train_label'] = _load_label(key_file['test_label'])
  dataset['test_img'] = _load_img(key_file['test_img'])
  dataset['test_label'] = _load_label(key_file['test_label'])
  return dataset

def init_mnist():
  """
  初始化MNIST数据集：
  1.下载所有原始压缩文件（如未下载）
  2.将其转换为NumPy数组
  3.将所有数据保存为一个pickle文件，便于后续快速加载
  """
  download_mnist()
  dataset = _convert_numpy()
  print("Creating pickle file...")
  # 以二进制写模式打开pickle文件
  with open(save_file, 'wb') as f:
    # pickle.dump将Python对象序列化并写入文件，参数 -1 表示使用最高效的协议
    pickle.dump(dataset, f, -1)
  print("Done!")

def _change_one_hot_label(X):
  """
  将标签数组转换为one-hot编码格式
  参数：
    X：一维整数标签数组，形状为(N,)
  返回：
    T：二维数组，形状为(N, 10)，每行只有一个位置为1，其余为0
    例如标签5转换为 [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.]

  One-hot 编码是一种将分类变量（离散标签）转换为数值向量的方法，
  目的是让机器学习算法能够正确处理非数值型的类别信息，
  同时避免引入虚假的“大小关系”。
  向量长度 = 类别总数。
  每个向量中只有一位是 1，其余全是 0，“One-hot”即“独热”——仅有一个位置被激活  
  """

  # X 存放所有样本真实标签的一维数组[3, 7, 0, 2, ...]
  # T 全零二维数组，准备填充为 One-hot 矩阵
  # row 在循环中代表 T 的当前行（一个一维数组） # 
  # X[idx] 第 idx 个样本的真实标签（一个整数）	比如 3
  # np.zeros 是 NumPy 库中用于创建全零数组的核心函数。它生成一个指定形状和数据类型的新数组，所有元素的值初始化为 0（或 0.0）。
  T = np.zeros((X.size, 10))
  #enumerate(T) 返回一个迭代器，每次迭代生成一个元组 (索引, 元素)
  for idxm, row in enumerate(T):
    # 第idx个样本的标签未 X[idx],将该行对应列设置为1
    # row[x]这里面拿到的是：循环到的[0,0,0,0,0,0,0,0]
    # X[idxm]：idxm是循环到的索引，X是真实标签[3, 7, 0, 2, ...]
    # 如X[0] = 3,与之对应的T[0][3]=[0,0,0,1,0,0,0,0,0,0]
    row[X[idxm]] = 1
  return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
  """
  MNIST数据集的加载函数（主入口）

  参数：
  ----------
  normalize: bool
    若为True， 将图像像素值从[0, 255]归一化到[0.0, 1.0]

  flatten: bool
    若为True,图像数据为一维数组(784,);
    若为False,图像保持二维形状(1, 28, 28),适合CNN（卷积神经网络）输入

  one_hot_label: bool
  若为True，标签以one-hot编码形式返回
  若为False，标签为原始整数0~9

  返回：
  ----------
  （训练图像，训练标签），（测试图像，测试标签）
  """
  # 如果pickle文件不存在，先执行初始化流程（下载+转换+保存）
  if not os.path.exists(save_file):
    init_mnist()

  # 从pickle文件中加载已处理好的数据集字典
  with open(save_file, 'rb') as f:
    dataset = pickle.load(f)

  # 归一化：将像素值从0——255收缩到0.0~1.0
  if normalize:
    for key in ('train_img', 'test_img'):
      dataset[key] = dataset[key].astype(np.float32) #转换为浮点数
      dataset[key] /= 255.0

  # 是否转换为one-hot标签
  if one_hot_label:
    dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
    dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

  # 是否保持图像原始形状（不展平）
  if not flatten:
    # 重塑为（样本数，通道数，高，宽）的格式
    for key in ('train_img', 'test_img'):
      dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

  # 返回训练数据和测试数据元组
  return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

# 如果直接运行此脚步，则执行初始化操作（下载并生成pickle文件）
if __name__ == '__main__':
  init_mnist()

  