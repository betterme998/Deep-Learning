"""
文件是否存在
比较新旧两种检查方式
"""

import os
from pathlib import Path

target_path = "./README.md"

# 旧式：os.path 方式
if os.path.exists(target_path):
  print("文件存在") # 文件存在

# 新式：pathlib 方式 (推荐)
path_obj = Path(target_path)
print(f'Path返回类型：{type(path_obj)}') # ：<class 'pathlib.WindowsPath'>
print(f'Path返回值：{path_obj}') # Path返回值：README.md
if path_obj.exists():
  print("文件存在(pathlib)") # 文件存在(pathlib)