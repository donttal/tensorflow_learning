import tensorflow as tf
slim = tf.contrib.slim
import train_evaluation

def net_architecture(images, num_classes=10, is_training=False,
                     dropout_keep_prob=0.5,
                     spatial_squeeze=True,
                     scope='Net'):

    # Create empty dictionary
    end_points = {}

    with tf.variable_scope(scope, 'Net', [images, num_classes]) as sc:
        end_points_collection = sc.name + '_end_points'

        # Collect outputs for conv2d and max_pool2d.
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.max_pool2d], 
        outputs_collections=end_points_collection):
        
            # Layer-1
            net = tf.contrib.layers.conv2d(images, 32, [5,5], scope='conv1')
            net = tf.contrib.layers.max_pool2d(net, [2, 2], 2, scope='pool1')

            # Layer-2
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], scope='conv2')
            net = tf.contrib.layers.max_pool2d(net, [2, 2], 2, scope='pool2')

            # Layer-3
            net = tf.contrib.layers.conv2d(net, 1024, [7, 7], padding='VALID', scope='fc3')
            net = tf.contrib.layers.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout3')

            # Last layer which is the logits for classes
            logits = tf.contrib.layers.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='fc4')

            # Return the collections as a dictionary
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            # Squeeze spatially to eliminate extra dimensions.
            if spatial_squeeze:
                logits = tf.squeeze(logits, [1, 2], name='fc4/squeezed')
                end_points[sc.name + '/fc4'] = logits
            return logits, end_points

def net_arg_scope(weight_decay=0.0005):
    #Defines the default network argument scope.

    with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.conv2d],
            padding='SAME',
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                               uniform=False, seed=None,
                                                                               dtype=tf.float32),
            activation_fn=tf.nn.relu) as sc:
        return sc

graph = tf.Graph()
with graph.as_default():
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
    # Place holders
    image_place = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='image')
    label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name='gt')
    dropout_param = tf.placeholder(tf.float32)

        # MODEL
    arg_scope = net.net_arg_scope(weight_decay=0.0005)
    with tf.contrib.framework.arg_scope(arg_scope):
        logits, end_points = net.net_architecture(image_place, num_classes=FLAGS.num_classes, dropout_keep_prob=dropout_param,
                                       is_training=FLAGS.is_training)
    # Define loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_place))
    # Accuracy
    with tf.name_scope('accuracy'):
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_place, 1))
        # Accuracy calculation
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
    # Define optimizer by its default values
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Gradient update.
    with tf.name_scope('train'):
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    arr = np.random.randint(data.train.images.shape[0], size=(3,))
    tf.summary.image('images', data.train.images[arr], max_outputs=3,
                     collections=['per_epoch_train'])

    # Histogram and scalar summaries sammaries
    for end_point in end_points:
        x = end_points[end_point]
        tf.summary.scalar('sparsity/' + end_point,
                          tf.nn.zero_fraction(x), collections=['train', 'test'])
        tf.summary.histogram('activations/' + end_point, x, collections=['per_epoch_train'])

    # Summaries for loss, accuracy, global step and learning rate.
    tf.summary.scalar("loss", loss, collections=['train', 'test'])
    tf.summary.scalar("accuracy", accuracy, collections=['train', 'test'])
    tf.summary.scalar("global_step", global_step, collections=['train'])
    tf.summary.scalar("learning_rate", learning_rate, collections=['train'])

    # Merge all summaries together.
    summary_train_op = tf.summary.merge_all('train')
    summary_test_op = tf.summary.merge_all('test')
    summary_epoch_train_op = tf.summary.merge_all('per_epoch_train')

tensors_key = ['cost', 'accuracy', 'train_op', 'global_step', 'image_place', 'label_place', 'dropout_param',
                   'summary_train_op', 'summary_test_op', 'summary_epoch_train_op']
tensors = [loss, accuracy, train_op, global_step, image_place, label_place, dropout_param, summary_train_op,
               summary_test_op, summary_epoch_train_op]
tensors_dictionary = dict(zip(tensors_key, tensors))

# Configuration of the session
session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
sess = tf.Session(graph=graph, config=session_conf)

with sess.as_default():
    # Run the saver.
    # 'max_to_keep' flag determines the maximum number of models that the tensorflow save and keep. default by TensorFlow = 5.
    saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoint)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    ###################################################
    ############ Training / Evaluation ###############
    ###################################################
    train_evaluation.train(sess, saver, tensors_dictionary, data,
                             train_dir=FLAGS.train_dir,
                             finetuning=FLAGS.fine_tuning,
                             num_epochs=FLAGS.num_epochs, checkpoint_dir=FLAGS.checkpoint_dir,
                             batch_size=FLAGS.batch_size)
                                 
    train_evaluation.evaluation(sess, saver, tensors_dictionary, data,
                           checkpoint_dir=FLAGS.checkpoint_dir)