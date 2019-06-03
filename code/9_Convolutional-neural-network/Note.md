# 模型实现细节
参考该文章进行编写
http://machinelearninguru.com/deep_learning/tensorflow/neural_networks/cnn_classifier/cnn_classifier.html
dropout层在行'29'中定义。所述dropout_keep_prob参数确定的神经元的部分，其保持不变，并不会由漏失层禁用。此外，标志is_training应该激活并停用丢失层，这会迫使丢失在训练阶段处于活动状态并停用 它在测试/评估阶段。