# This code create some arbitrary variables and initialize them ###
# The goal is to show how to define and initialize variables from scratch.
import tensorflow as tf
from tensorflow.python.framework import ops

#######################################
######## Defining Variables ###########
#######################################

# Create three variables with some default values.
weights = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name="weights")
biases = tf.Variable(tf.zeros([3]), name="biases")
custom_variable = tf.Variable(tf.zeros([3]), name="custom")

# Get all the variables' tensors and store them in a list.
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
'''
初始化
Initializers必须在模型中的所有其他操作之前运行变量。
作为类比，我们可以考虑汽车的驱动器。
变量也可以来自保存的模型，例如检查点文件
变量可以全局初或者从其他变量初始化。
'''
############################################
######## Customized initializer ############
############################################

## Initialation of some custom variables.
## In this part we choose some variables and only initialize them rather than initializing all variables.

# "variable_list_custom" is the list of variables that we want to initialize.
variable_list_custom = [weights, custom_variable]

# The initializer
init_custom_op = tf.variables_initializer(var_list=variable_list_custom)

########################################
######## Global initializer ############
########################################

# Method-1
# Add an op to initialize the variables.
init_all_op = tf.global_variables_initializer()

# Method-2
init_all_op = tf.variables_initializer(var_list=all_variables_list)

##########################################################
######## Initialization using other variables ############
##########################################################

# 通过使用initialized_value（）获取值，可以使用其他现有变量的初始值初始化新变量
WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")

# Now, the variable must be initialized.
init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])

######################################
####### Running the session ##########
######################################
# 到目前为止我们所做的只是定义初始化器的操作并将它们放在图表上。
# 为了真正初始化变量，必须在会话中运行已定义的初始化程序的操作。脚本如下：
with tf.Session() as sess:
    # Run the initializer operation.
    sess.run(init_all_op)
    sess.run(init_custom_op)
    sess.run(init_WeightsNew_op)
