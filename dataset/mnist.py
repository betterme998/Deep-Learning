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