"""
获取当前脚本所在目录的绝对路径。
常用来设置项目根目录、定位配置文件、拼接资源路径等。
"""
from pathlib import Path
# Path: pathlib.Path 类，用于创建路径对象
# __file__:是当前脚本文件的路径（可能是相对路径）
# Path(__file__) 实例化了一个指向该脚本文件的路径对象
p = Path(__file__)  # D:\代码\Deep-Learning\1_python\09_获取当前脚本绝对路径.py
print(f'该脚本文件的路径:{p}')

# 路径对象的 resolve() 方法返回一个新的绝对路径对象
absolute_path = p.resolve()
print(f'该脚本文件的绝对路径:{absolute_path}') #D:\代码\Deep-Learning\1_python\09_获取当前脚本绝对路径.py

# 路径对象的 parent 属性返回该路径的直接父目录的路径对象
BASE_DIR = absolute_path.parent
print(f'该脚本文件的父目录:{BASE_DIR}') # D:\代码\Deep-Learning\1_python
BASE_DIR = Path(__file__).resolve().parent