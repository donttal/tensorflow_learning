from __future__ import print_function
import tensorflow as tf
import os

log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs'
# Defining some sentence!
welcome = tf.constant('Welcome to TensorFlow world!')

# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(log_dir), sess.graph)
    print("output: ", sess.run(welcome))

# Closing the writer.
writer.close()
sess.close()
