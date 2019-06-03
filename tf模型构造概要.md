# tensorflow定义参数的一般格式

tf.app.flags.DEFINE_integer('num_epochs', 50, 'The number of epochs for training the model. Default=50')

##### Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS
##### 在下文就可以直接引用变量
for epoch_num in range(FLAGS.num_epochs): 

# tensorflow模型构建

一个比较简单的构建流程就是使用函数定义

## 数据库的分析

### Dataset peparation ###
```python
# Dataset loading and organizing.
iris = datasets.load_iris()
# Only the first two features are extracted and used.
X = iris.data[:, :2]
# The labels are transformed to -1 and 1.
y = np.array([1 if label==0 else -1 for label in iris.target])
# Get the indices for train and test sets.
my_randoms = np.random.choice(X.shape[0], X.shape[0], replace=False)
train_indices = my_randoms[0:int(0.5 * X.shape[0])]
test_indices = my_randoms[int(0.5 * X.shape[0]):]
# Splitting train and test sets.
x_train = X[train_indices]
y_train = y[train_indices]
x_test = X[test_indices]
y_test = y[test_indices]
```

## 定义全局变量/加入agrese参数，命令行修改模型参数
```python
# creating the weight and bias.
# The defined variables will be initialized to zero.
W = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")
```

## 定义模型的输入

```python
# Creating placeholders for input X and label Y.

def inputs():
    """
    Defining the place_holders.
    :return:
            Returning the data and label place holders.
    """
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    return X,Y
```

## 计算模型预测
```python
# Create the prediction.

def inference(X):
    """
    Forward passing the X.
    :param X: Input.
    :return: X*W + b.
    """
    return X * W + b
```
## 计算模型损失
```python
def loss(X, Y):
    '''
    compute the loss by comparing the predicted value to the actual label.
    :param X: The input.
    :param Y: The label.
    :return: The loss over the samples.
    '''

    # Making the prediction.
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))/(2*data.shape[0])
```


## 定义模型优化器

```python
def train(loss):
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```

## 建立会话Session
```python 
with tf.Session() as sess:

    # Initialize the variables[w and b].
    sess.run(tf.global_variables_initializer())
    
    # Get the input tensors
    X, Y = inputs()
    
    # Return the train loss and create the train_op.
    train_loss = loss(X, Y)
    train_op = train(train_loss)
    
    for epoch_num in range(FLAGS.num_epochs): # run 100 epochs
        loss_value, _ = sess.run([train_loss,train_op],
                                 feed_dict={X: data[:,0], Y: data[:,1]})
    
        # Displaying the loss per epoch.
        print('epoch %d, loss=%f' %(epoch_num+1, loss_value))
    
        # save the values of weight and bias
        wcoeff, bias = sess.run([W, b])
```

## 建立会话的另一种写法
```python
# Defining Placeholders

x_data = tf.placeholder(shape=[None, X.shape[1]], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
W = tf.Variable(tf.random_normal(shape=[X.shape[1],1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Calculation of loss and accuracy.

total_loss = loss_fn(W, b, x_data, y_target)
accuracy = inference_fn(W, b, x_data, y_target)

# Defining train_op

train_op = tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate).minimize(total_loss)

### Session
sess = tf.Session() ###
## Initialization of the variables.
init = tf.initialize_all_variables()
sess.run(init)
将所有的计算过程定位成为全局变量，最后使用tensorflow来进行初始化

for step_idx in range(FLAGS.num_steps):

    # Get the batch of data.
    X_batch, y_batch = next_batch_fn(x_train, y_train, num_samples=FLAGS.batch_size)
    
    # Run the optimizer.
    sess.run(train_op, feed_dict={x_data: X_batch, y_target: y_batch})
    
    # Calculation of loss and accuracy.
    loss_step = sess.run(total_loss, feed_dict={x_data: X_batch, y_target: y_batch})
    train_acc_step = sess.run(accuracy, feed_dict={x_data: x_train, y_target: np.transpose([y_train])})
    test_acc_step = sess.run(accuracy, feed_dict={x_data: x_test, y_target: np.transpose([y_test])})
    
    # Displaying the desired values.
    if step_idx % 100 == 0:
        print('Step #%d, training accuracy= %% %.2f, testing accuracy= %% %.2f ' % (step_idx, float(100 * train_acc_step), float(100 * test_acc_step)))
```

## 可视化训练结果
```python
Input_values = data[:,0]
Labels = data[:,1]
Prediction_values = data[:,0] * wcoeff + bias

# # uncomment if plotting is desired!
# plt.plot(Input_values, Labels, 'ro', label='main')
# plt.plot(Input_values, Prediction_values, label='Predicted')

# # Saving the result.
# plt.legend()
# plt.savefig('plot.png')
# plt.close()
```
快速查找相关函数的网址
https://www.w3cschool.cn/tensorflow_python/tf_keras_backend_cast.html


## 逻辑回归类别的例子
```python
# 准确率计算
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
```

# 自调整学习率的tf实现
```python
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 1, 'Number of epoch pass to decay learning rate.')

    # global step
    global_step = tf.Variable(0, name="global_step", trainable=False)
    
    # learning rate policy
    decay_steps = int(num_train_samples / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               FLAGS.learning_rate_decay_factor,
                                               staircase=True,
                                               name='exponential_decay_learning_rate')
```
# 使用checkpoint来保存模型
```python
    with sess.as_default():
    
        # The saver op.
        saver = tf.train.Saver()
        # The prefix for checkpoint files
        checkpoint_prefix = 'model'
        ######
        模型训练过程
        ######
        ###########################################################
        ############ Saving the model checkpoint ##################
        ###########################################################
    
        # # The model will be saved when the training is done.
    
        # Create the path for saving the checkpoints.
        if not os.path.exists(FLAGS.checkpoint_path):
            os.makedirs(FLAGS.checkpoint_path)
    
        # save the model
        save_path = saver.save(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
        print("Model saved in file: %s" % save_path)
    
        ############################################################################
        ########## Run the session for pur evaluation on the test data #############
        ############################################################################
    
        # The prefix for checkpoint files
        checkpoint_prefix = 'model'
    
        # Restoring the saved weights.重新加载之前的模型权重
        saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
        print("Model restored...")
    
        # Evaluation of the model
        test_accuracy = 100 * sess.run(accuracy, feed_dict={
            image_place: data['test/image'],
            label_place: data['test/label'],
            dropout_param: 1.})
    
        print("Final Test Accuracy is %% %.2f" % test_accuracy)
    
        参考网址保存模型
        https://www.jianshu.com/p/b0c789757df6
```