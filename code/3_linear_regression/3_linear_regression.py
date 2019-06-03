'''
使用TensorFlow实现线性回归
通过TensorFlow训练线性模型以适应数据。
在机器学习和统计中，线性回归是对因变量Y和至少一个自变量X之间的关系的建模。
在线性回归中，线性关系将由预测函数建模，其参数将通过数据进行估计。
线性回归算法的主要优点是其简单性，使用它可以非常直接地解释新模型并将数据映射到新空间。
代码主要是实现使用TensorFLow训练线性模型以及如何展示生成的模型。
该代码数据集合是随机生成，实际开发的时候需要特征工程处理原始数据集
'''
import numpy as np
import tensorflow as tf
import xlrd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from sklearn.utils import check_random_state

# Generating artificial data.
n = 50
XX = np.arange(n)
rs = check_random_state(0)
YY = rs.randint(-20, 20, size=(n, )) + 2.0 * XX
data = np.stack([XX, YY], axis=1)

#######################
## Defining flags #####
#######################
# 在命令行运行该python文件中可以输入python linear_regression.py --num_epochs=100来修改num_epochs的数量
tf.app.flags.DEFINE_integer(
    'num_epochs', 50,
    'The number of epochs for training the model. Default=50')
# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# creating the weight and bias.
# The defined variables will be initialized to zero.
W = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")


#  Creating placeholders for input X and label Y.
def inputs():
    """
    Defining the place_holders.
    :return:
            Returning the data and label place holders.
    """
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    return X, Y


# Create the prediction.
def inference(X):
    """
    Forward passing the X.
    :param X: Input.
    :return: X*W + b.
    """
    return X * W + b


def loss(X, Y):
    '''
    compute the loss by comparing the predicted value to the actual label.
    :param X: The input.
    :param Y: The label.
    :return: The loss over the samples.
    '''

    # Making the prediction.
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(
        Y, Y_predicted)) / (2 * data.shape[0])


# The training function.
def train(loss):
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:

    # Initialize the variables[w and b].
    sess.run(tf.global_variables_initializer())

    # Get the input tensors
    X, Y = inputs()

    # Return the train loss and create the train_op.
    train_loss = loss(X, Y)
    train_op = train(train_loss)

    # Step 8: train the model
    for epoch_num in range(FLAGS.num_epochs):  # run 50 epochs
        loss_value, _ = sess.run([train_loss, train_op],
                                 feed_dict={
                                     X: data[:, 0],
                                     Y: data[:, 1]
                                 })

        # Displaying the loss per epoch.
        print('epoch %d, loss=%f' % (epoch_num + 1, loss_value))

        # save the values of weight and bias
        wcoeff, bias = sess.run([W, b])

###############################
#### Evaluate and plot ########
###############################
Input_values = data[:, 0]
Labels = data[:, 1]
Prediction_values = data[:, 0] * wcoeff + bias

# uncomment if plotting is desired!
plt.plot(Input_values, Labels, 'ro', label='main')
plt.plot(Input_values, Prediction_values, label='Predicted')

# Saving the result.
plt.legend()
plt.savefig('plot.png')
plt.close()