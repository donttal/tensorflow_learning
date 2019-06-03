# 导入所需的库
from __future__ import print_function
import tensorflow as tf
import os

# 在os.path.dirname(os.path.abspath(__file__))获取当前Python文件的目录名
log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs'
# Defining some sentence!
welcome = tf.constant('Welcome to TensorFlow world!')

# Run the session
# 在tf.summary.FileWriter被定义为摘要写入event files的.
# sess.run()必须被用于任何评价Tensor，否则操作将不被执行。
# 最后通过使用writer.close()，摘要编写器将被关闭。
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(log_dir), sess.graph)
    print("output: ", sess.run(welcome))

# Closing the writer.
writer.close()
sess.close()
