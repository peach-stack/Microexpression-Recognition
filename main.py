import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


# 查询文件
dir_name = 'data'
files = os.listdir(dir_name)
file_path = dir_name + os.sep + files[0]
print(file_path)

# 提取fer2013中的数据
data = pd.read_csv(file_path)
labels = np.array(data['emotion'])
pixels = np.array(data['pixels'])
# 总数据量35000+，这里只选择部分进行训练
N_sample = labels.size
# 先创建矩阵用于以后将像素变成N_sample*48*48的像素矩阵存储照片使用
# 并存储对应的标签
Face_data = np.zeros((N_sample, 48 * 48))
Face_label = np.zeros((N_sample, 7), dtype=int)

# 数据读取
for i in range(N_sample):
    x = pixels[i]
    x = np.fromstring(x, dtype=float, sep=' ')
    Face_data[i] = x
    Face_label[i, int(labels[i])] = 1
#    if i < 10:
#       print('i: %d \t ' % i, Face_data[i])


# 数据参数初始化
train_num = 30000
test_num = 5000
batch_size = 50
train_x = Face_data[0:train_num]
train_y = Face_label[0:train_num]
test_x = Face_data[train_num: train_num + test_num]
test_y = Face_label[train_num: train_num + test_num]
train_batch_num = train_num / batch_size
test_batch_num = test_num / batch_size

# 模型参数初始化
train_epoch = 100   # 训练数据的迭代次数
learning_rate = 0.001    # 学习率
n_input = 2304  # 输入维度
n_classes = 7  # 种类
dropout = 0.5  # 丢失率，防止过拟合
print("Prepared")

# 权重
weights = {
    # 卷积核
    # 3x3 conv, 1 input, 128 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 128], stddev=0.1)),
    # 3x3 conv, 128 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=0.1)),
    # 3x3 conv, 64 inputs, 32 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=0.1)),
    # fully connected,
    'w_fc': tf.Variable(tf.random_normal([6 * 6 * 32, 200])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([200, n_classes]))
}
# 偏置量
biases = {
    'bc1': tf.Variable(tf.random_normal([128])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([32])),
    'b_fc': tf.Variable(tf.random_normal([200])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# 设置计算图的输入节点
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# 卷积层
def conv2d(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# pooling层
def maxpool2d(x, k):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, [1, k, k, 1], [1, k, k, 1], padding='VALID')


# 卷积神经网络
def conv_net(x, weights, biases, dropout):
    # 把tensor变成四维矩阵
    x = tf.reshape(x, shape=[-1, 48, 48, 1])
    # 卷积层和池化层
    # 第一层
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    # 第二层
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    # 第三层
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
    # 全连接层，这边需要将池化后的结果对接全连接层
    fc1 = tf.reshape(conv3, [-1, weights['w_fc'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w_fc']), biases['b_fc'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# 绘制损失和精确度图像
def make_plot(Test_acc, Test_loss):
    #x_coordinate = np.linspace(0, 1, epoch)
    # 画损失函数迭代图像
    plt.subplot(2, 1, 1)
    plt.ylabel('Test loss')
    plt.title('Loss')
    plt.plot(Test_loss, 'r')
    # 画精确度迭代图像
    plt.subplot(2, 1, 2)
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy')
    plt.plot(Test_acc, 'g')
    plt.show()



# 构建计算图及相关的模型
# 创建模型
pred = conv_net(x, weights, biases, keep_prob)
# 损失函数和优化器定义
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# 评估参数定义
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 初始化参数变量
Train_ind = np.arange(train_num)
Test_ind = np.arange(test_num)
# 存储每次epoch最终的结果参数用于最后画图
Test_loss = []
Test_acc = []


# 运行Session
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(max_to_keep=1)    # 保存最后一代模型
    for epoch in range(0, train_epoch):
        Total_test_loss = 0
        Total_test_acc = 0
        for train_batch in range(0, int(train_batch_num)):
            sample_ind = Train_ind[train_batch * batch_size: (train_batch + 1) * batch_size]
            batch_x = train_x[sample_ind, :]
            batch_y = train_y[sample_ind, :]

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            # 每测试一大组数据就把最新训练的数据在带入检验
            # if train_batch % 100 == 0:
            #     loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
            #                                                       y: batch_y,
            #                                                       keep_prob: 1.})
            #     print("Epoch: " + str(epoch + 1)
            #           + ", Batch: " + str(train_batch)
            #           + ", Loss= {:.3f}".format(loss)
            #           + ", Training Accuracy= {:.3f}".format(acc))

        # 每进行一次epoch后将用5000个test进行检测得到准确率和损失函数图像
        for test_batch in range(0, int(test_batch_num)):
            sample_ind = Test_ind[test_batch * batch_size:(test_batch + 1) * batch_size]
            batch_x = test_x[sample_ind, :]
            batch_y = test_y[sample_ind, :]
            test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                        y: batch_y,
                                                                        keep_prob: 1.})
            Total_test_lost = Total_test_loss + test_loss
            Total_test_acc = Total_test_acc + test_acc

        Total_test_acc = Total_test_acc / test_batch_num
        Total_test_loss = Total_test_lost / test_batch_num
        Test_loss.append(Total_test_loss)
        Test_acc.append(Total_test_acc)
        print("Epoch: " + str(epoch + 1)
              + ", Test Loss = {:.8f}".format(Total_test_loss)
              + ", Test Accuracy = {:.8f}".format(Total_test_acc))
    saver.save(sess, "save/model")
make_plot(Test_acc, Test_loss)
print("Finished")
