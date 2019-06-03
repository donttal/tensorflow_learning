# Multi Layer Perceptron
此代码用于培训Multi Layer Perceptron架构，其中输入将被转发到包含一些隐藏层的网络。
# Training
可以使用train.sh bash脚本文件使用以下命令运行训练：
```
./train.sh
```
The bash script is as below:
```
python train_mlp.py \
  --batch_size=512 \
  --max_num_checkpoint=10 \
  --num_classes=10 \
  --num_epochs=1 \
  --initial_learning_rate=0.001 \
  --num_epochs_per_decay=1 \
  --is_training=True \
  --allow_soft_placement=True \
  --fine_tuning=False \
  --online_test=True \
  --log_device_placement=False
```
helper:
为了实现运行以下命令的输入参数是什么:
```
python train_mlp.py --help
```
其中train_mlp.py是运行培训的主要文件。上述命令的结果如下：
```python
--train_dir TRAIN_DIR
                      写入事件日志的目录。
--checkpoint_dir CHECKPOINT_DIR
                      写入检查点的目录。
--max_num_checkpoint MAX_NUM_CHECKPOINT
                      TensorFlow的最大检查点数
                      保持。
--num_classes NUM_CLASSES
                      要部署的模型克隆数。
--batch_size BATCH_SIZE
                      要部署的模型克隆数。
--num_epochs NUM_EPOCHS
                      时代数的训练。
--initial_learning_rate INITIAL_LEARNING_RATE
                      初始学习率。
--learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR
                      学习率衰减因子。
--num_epochs_per_decay NUM_EPOCHS_PER_DECAY
                      时代传递到衰减学习率的数量。
--is_training [IS_TRAINING]
                      训练/测试。
--fine_tuning [FINE_TUNING]
                      是否需要微调？。
--online_test [ONLINE_TEST]
                      是否需要微调？。
--allow_soft_placement [ALLOW_SOFT_PLACEMENT]如果没有，则
                      自动将变量放在CPU上
                      GPU支持。
--log_device_placement [LOG_DEVICE_PLACEMENT]
                      演示哪些变量在哪个设备上。
```

# Evaluation
将使用以下命令使用evaluation.sh bash脚本文件运行评估:
```
./evaluation.sh
```