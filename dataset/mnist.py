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