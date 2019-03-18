# tf.contrib.slim学习

TF-Slim是一个轻量级库，用于在TensorFlow中定义，训练和评估复杂模型。可自由的把tf-slim与tensorflow或其他框架组合使用；

可以非常方便的使用TF-slim的variables, layers 和scopes定义复杂的网络模型。

[TOC]



## Variable

相比tensorflow的Variable的定义，TF-slim在定义变量时增加了**将变量加入正则化loss**中和**指定初始化设备**的参数

```python
weights = slim.variable('weights',
                             shape=[10, 10, 3 , 3],
                             initializer=tf.truncated_normal_initializer(stddev=0.1),
                             regularizer=slim.l2_regularizer(0.05),
                             device='/CPU:0')
```

同时，TF-slim 还可以区分**模型变量**和**非模型变量**，模型变量：model在inference是需要用到的变量称为模型变量

非模型变量：其余的为非模型变量，如global_step

```python
# Model Variables
weights = slim.model_variable('weights',
                              shape=[10, 10, 3 , 3],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05),
                              device='/CPU:0')
model_variables = slim.get_model_variables()
 
# Regular variables
my_var = slim.variable('my_var',
                       shape=[20, 1],
                       initializer=tf.zeros_initializer())
regular_variables_and_model_variables = slim.get_variables()
```

TF-slim会自动把在TF-slim‘s layers中定义的变量和通过slim.model_variable()创建的变量加入tf.GraphKeys.MODEL-VARIABLES collection中，也可指定变量为模型变量；

```python
my_model_variable = CreateViaCustomCode()
 
# Letting TF-Slim know about the additional variable.
slim.add_model_variable(my_model_variable)
```

## layers

(1)  对于一些有固定流程的网络层，TF-slim把它封装成更高级的层，如卷积层的流程：

- 创建weight和bias变量

- 用上层的输出和weight做卷积

- 加上bias

- 激活

  ```
  ## input = ...
  with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
  ```

  而TF-slim实现conv2d一句话就够了

  ```
  input = ...
  net = slim.conv2d(input, 128, [3, 3], scope='conv1_1')
  ```

  

TF-slim提供了**repea**t和**stack**两个进行重复操作的函数

```
net = ...
net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
```

TF-slim实现上面的三个卷积层

```
net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
```

TF-slim不仅可以实现重复操作，还会只能的为不同的scope命名；如上面的slim.repeat给这三个卷积层命名为conv3/conv3_1,conv3/conv3_2,conv3/conv3_3;



TF-Slim的slim.stack运算符允许调用者使用不同的参数重复应用相同的操作来创建层塔。 slim.stack还为每个创建的操作创建一个新的tf.variable_scope。例如，创建多层感知器（MLP）的简单方法

```
# Verbose way:
x = slim.fully_connected(x, 32, scope='fc/fc_1')
x = slim.fully_connected(x, 64, scope='fc/fc_2')
x = slim.fully_connected(x, 128, scope='fc/fc_3')
 
# Equivalent, TF-Slim way using slim.stack:
slim.stack(x, slim.fully_connected, [32, 64, 128], scope='fc')
```

使用TF-slim创建卷积核大小和数量不同的卷积层

```python
# Verbose way:
x = slim.conv2d(x, 32, [3, 3], scope='core/core_1')
x = slim.conv2d(x, 32, [1, 1], scope='core/core_2')
x = slim.conv2d(x, 64, [3, 3], scope='core/core_3')
x = slim.conv2d(x, 64, [1, 1], scope='core/core_4')
 
# Using stack:
slim.stack(x, slim.conv2d, [(32, [3, 3]), (32, [1, 1]), (64, [3, 3]), (64, [1, 1])], scope='core')
```

##  scope

tensorflow有范围机制**(name_scope,variable_scope)**,除此之外，TF_slim增加一个新的范围机制**arg_scope**

一个为参数设置默认值的范围机制；

先看一段三个卷积层的定义：

```
net = slim.conv2d(inputs, 64, [11, 11], 4, padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
net = slim.conv2d(net, 128, [11, 11], padding='VALID',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
net = slim.conv2d(net, 256, [11, 11], padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')
```

 这段代码不易阅读，存在大量的重复设置的参数，一种解决的方案是为这些变量指定默认的值；

```
padding = 'SAME'
initializer = tf.truncated_normal_initializer(stddev=0.01)
regularizer = slim.l2_regularizer(0.0005)
net = slim.conv2d(inputs, 64, [11, 11], 4,
                  padding=padding,
                  weights_initializer=initializer,
                  weights_regularizer=regularizer,
                  scope='conv1')
net = slim.conv2d(net, 128, [11, 11],
                  padding='VALID',
                  weights_initializer=initializer,
                  weights_regularizer=regularizer,
                  scope='conv2')
net = slim.conv2d(net, 256, [11, 11],
                  padding=padding,
                  weights_initializer=initializer,
                  weights_regularizer=regularizer,
                  scope='conv3')
```

但是这样写代码依然很混乱；通过使用arg_scope会为scope设定默认值，并简化代码

```
  with slim.arg_scope([slim.conv2d], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.conv2d(inputs, 64, [11, 11], scope='conv1')
    net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='conv2')
    net = slim.conv2d(net, 256, [11, 11], scope='conv3')
```

arg_scope指定的变量的默认值可以在本地被重写，同时arg_scope也可以嵌套使用；

```python
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
  with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
    net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
    net = slim.conv2d(net, 256, [5, 5],
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
                      scope='conv2')
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc')
```

实际例子

```
def vgg16(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
  return net
```

