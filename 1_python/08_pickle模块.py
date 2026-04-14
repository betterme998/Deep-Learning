"""
pickle 是 Python 标准库中的一个模块，用于实现对象的序列化（serialization）和反序列化（deserialization）。
主要功能
序列化：将 Python 对象（如列表、字典、自定义类实例等）转换为字节流，以便存储到文件或通过网络传输。
反序列化：从字节流中还原出原来的 Python 对象。

pickle.dump(obj, file)	将对象 obj 序列化并写入到文件对象 file 中
pickle.load(file)	从文件对象 file 中读取字节流并反序列化为 Python 对象
pickle.dumps(obj)	将对象序列化为字节串（bytes），不写入文件
pickle.loads(bytes)	从字节串反序列化出对象

"""
import pickle

# 要保存的数据
data = {"name": "Alice", "scores": [98,97,92]}

# 序列化到文件
with open("data.pkl", "wb") as f: # wb表示以二进制形式打开文件 w代表写入模式 b代表二进制格式操作
  pickle.dump(data, f)

# 从文件反序列化
with open("data.pkl", "rb") as f: # rb表示以二进制形式打开文件 r代表读取模式 b代表二进制格式操作
  data_loaded = pickle.load(f)

print(data_loaded) #{"name": "Alice", "scores": [98,97,92]}