## tensorflow中arg_scope的作用
When defining convolution layers, you may always use the same padding type and the same initializer, and maybe even the same convolution size. For you pooling, maybe you are also always using the same 2x2 pooling size. And so on.
arg_scope is a way to avoid repeating providing the same arguments over and over again to the same layer types.

1.Example of how to use tf.contrib.framework.arg_scope:

```python
from third_party.tensorflow.contrib.layers.python import layers
  arg_scope = tf.contrib.framework.arg_scope
  with arg_scope([layers.conv2d], padding='SAME',
                 initializer=layers.variance_scaling_initializer(),
                 regularizer=layers.l2_regularizer(0.05)):
    net = layers.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
    net = layers.conv2d(net, 256, [5, 5], scope='conv2')
```


The first call to conv2d will behave as follows:

```python
   layers.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                  initializer=layers.variance_scaling_initializer(),
                  regularizer=layers.l2_regularizer(0.05), scope='conv1')     
```

The second call to conv2d will also use the arg_scope’s default for padding:

```
  layers.conv2d(inputs, 256, [5, 5], padding='SAME',
                  initializer=layers.variance_scaling_initializer(),
                  regularizer=layers.l2_regularizer(0.05), scope='conv2')
```


Example of how to reuse an arg_scope:

```
with arg_scope([layers.conv2d], padding='SAME',
                 initializer=layers.variance_scaling_initializer(),
                 regularizer=layers.l2_regularizer(0.05)) as sc:
    net = layers.conv2d(net, 256, [5, 5], scope='conv1')

with arg_scope(sc):
    net = layers.conv2d(net, 256, [5, 5], scope='conv2')
```

