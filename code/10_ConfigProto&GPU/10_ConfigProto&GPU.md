# tensorflow中ConfigProto&GPU学习

# tensorflow ConfigProto

`tf.ConfigProto`一般用在创建`session`的时候。用来对`session`进行参数配置

```python
with tf.Session(config = tf.ConfigProto(...),...)
```

```python
#tf.ConfigProto()的参数
log_device_placement=True : 是否打印设备分配日志
allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
```

allow_soft_placement标志，允许不同的设备之间的切换回的往复。当用户将“GPU”分配给所有操作而不考虑使用TensorFlow的GPU实际上不支持所有操作时，这非常有用。在这种情况下，如果禁用*allow_soft_placement*运算符，则可能会显示错误，并且用户必须启动调试过程，但使用该标志可以通过自动从不受支持的设备切换到受支持的设备来防止此问题。

log_device_placement标志用于显示在哪些设备上设置的操作。这对于调试很有用，它会在终端中投射一个详细的对话框。

## 控制GPU资源使用率

```python
#allow growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
#内存，所以会导致碎片
```

```python
# per_process_gpu_memory_fraction
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config, ...)
#设置每个GPU应该拿出多少容量给进程使用，0.4代表 40%
```

## 控制使用哪块GPU

```
# 命令行运行
~/ CUDA_VISIBLE_DEVICES=0  python your.py#使用GPU0
~/ CUDA_VISIBLE_DEVICES=0,1 python your.py#使用GPU0,1
#注意单词不要打错

#或者在 程序开头
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
```

