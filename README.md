# tensorflow_learning
参考网上的tf1.X教程以及一部分书籍，希望能够使得tensorflow学习加快并且能够及时复习。在后续我会整理tf2.0的实验版本。
# Mac运行错误
用Mac第一次使用python 的第三方包 Matplotlib报错：
```
Python is not installed as a framework. The Mac OS X backend will not......
```
#### **解决方法**
在目录：~/.matplotlib 下创建一个文件: matplotlibrc**
输入：backend:TkAgg
保存即可

# 安装脚本

运行同目录下installEnv.sh文件，能够安装anaconda以及创建python3.6的环境，激活环境安装所依赖的包。

