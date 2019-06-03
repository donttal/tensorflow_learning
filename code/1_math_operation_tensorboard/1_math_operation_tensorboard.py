from __future__ import print_function

import os

import tensorflow as tf

'''
由于我们的目标是使用Tensorboard，我们需要一个目录来存储信息（操作及其相应的输出）。
此信息导出为event files。
文件可以转换为可视数据，以便用户能够评估体系结构和操作。
'''
# The default path for saving event files is the same folder of this python file.
tf.app.flags.DEFINE_string(
    'log_dir',
    os.path.dirname(os.path.abspath(__file__)) + '/logs',
    'Directory where event logs are written to.')
'''
在os.path.dirname(os.path.abspath(__file__))获取当前Python文件的目录名。
tf.app.flags.FLAGS使用FLAGS指标指向所有已定义标志的点。
从现在开始，可以使用标记FLAGS.log_dir来调用。
'''
FLAGS = tf.app.flags.FLAGS

'''
提示用户输入绝对路径
# os.path.expanduser is leveraged to transform '~' sign to the corresponding path indicator.
#       Example: '~/logs' equals to '/home/username/logs'
'''
if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError('You must assign absolute path for --log_dir')

# 定义常量值
a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

# 基本的数据运算
# https://blog.csdn.net/weixin_34360651/article/details/86796535
# name属性为了更好的Tensorboard可视化
x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")

# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir),
                                   sess.graph)
    print("a =", sess.run(a))
    print("b =", sess.run(b))
    print("a + b =", sess.run(x))
    print("a/b =", sess.run(y))

# Closing the writer.
writer.close()
sess.close()

# 可视化结构查看同目录note.md
# 或者运行：[tensorboard --logdir="absolute/path/to/log_dir"]

