# 使用h5py库读写超过内存的大数据
在简单数据的读操作中，我们通常一次性把数据全部读入到内存中。读写超过内存的大数据时，有别于简单数据的读写操作，受限于内存大小，通常需要指定位置、指定区域读写操作，避免无关数据的读写。
h5py库刚好可以实现这一功能

```python
import h5py
X= np.random.rand(100, 1000, 1000).astype('float32')
y = np.random.rand(1, 1000, 1000).astype('float32')

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('X_train', data=X)
h5f.create_dataset('y_train', data=y)
h5f.close()

# Load hdf5 dataset
h5f = h5py.File('data.h5', 'r')
X = h5f['X_train']
Y = h5f['y_train']
h5f.close()
```
# 主要操作路线
1. 打开文件头与文件中的数据头
2. 预留存储空间判断
3. 数据指定位置赋值
   
# 分块读写示例
```python
import sys
import h5py
import numpy as np


def save_h5(times=0):
    if times == 0:
        h5f = h5py.File('data.h5', 'w')
        dataset = h5f.create_dataset("data", (100, 1000, 1000),
                                     maxshape=(None, 1000, 1000),
                                     # chunks=(1, 1000, 1000),
                                     dtype='float32')
    else:
        h5f = h5py.File('data.h5', 'a')
        dataset = h5f['data']
    # 关键：这里的h5f与dataset并不包含真正的数据，
    # 只是包含了数据的相关信息，不会占据内存空间
    #
    # 仅当使用数组索引操作（eg. dataset[0:10]）
    # 或类方法.value（eg. dataset.value() or dataset.[()]）时数据被读入内存中
    a = np.random.rand(100, 1000, 1000).astype('float32')
    # 调整数据预留存储空间（可以一次性调大些）
    dataset.resize([times*100+100, 1000, 1000])
    # 数据被读入内存 
    dataset[times*100:times*100+100] = a
    # print(sys.getsizeof(h5f))
    h5f.close()

def load_h5():
    h5f = h5py.File('data.h5', 'r')
    data = h5f['data'][0:10]
    # print(data)

if __name__ == '__main__':
    # save_h5(0)
    for i in range(20):
        save_h5(i)
    # 部分数据导入
```