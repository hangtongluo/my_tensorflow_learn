# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:08:30 2017

@author: lht
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


'''Convolutional networks are more expensive computationally, 
so we'll limit its depth and number of fully connected nodes.'''

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()
with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(
            tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # 卷积核为64X64，in_size(in_channels)=1, outsize(output_channels)=16 :(把一个通道变成k个通道的)
    layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1)) #(5*5*1*16) 
    layer1_biases = tf.Variable(tf.zeros([depth]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1)) #(5*5*16*16)
    layer2_biases = tf.Variable(tf.constant(
            1.0, shape=[depth]))
    
    # layer3_weights = tf.Variable(tf.truncated_normal(
    #         [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1)) #(784,64)
    layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1)) #(7*7*16,64)
    

    layer3_biases = tf.Variable(tf.constant(
            1.0, shape=[num_hidden])) #64
    
    layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1)) #(64,10)
    layer4_biases = tf.Variable(tf.constant(
            1.0, shape=[num_labels])) #10

    keep_prob = tf.placeholder(tf.float32)

#==============================================================================
#     #Model
#     def model(data):
#         conv = tf.nn.conv2d(data, layer1_weights, [1,2,2,1], padding='SAME')
#         hidden = tf.nn.relu(conv + layer1_biases)
#         
#         conv = tf.nn.conv2d(hidden, layer2_weights, [1,2,2,1], padding='SAME')
#         hidden = tf.nn.relu(conv + layer2_biases)
#         
#         shape = hidden.get_shape().as_list()
#         reshape = tf.reshape(hidden,[shape[0], shape[1] * shape[2] * shape[3]])
#         hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#         
#         return tf.matmul(hidden, layer4_weights) + layer4_biases
#==============================================================================

        #Model
    conv2d_strides = [1,1,1,1]
    # conv2d_strides = [1,2,2,1] 

    pool_ksize = [1,2,2,1]
    # pool_ksize = [1,1,1,1]
    # pool_ksize = [1,3,3,1]

    pool_strides = [1,2,2,1]
    # pool_strides = [1,1,1,1]
    '''  # 卷积：
           strides：第1，第4参数都为1，中间两个参数为卷积步幅，如：[1,1,1,1],[1,2,2,1]
                 1、使用VALID方式,feature map的尺寸为       (3,3,1,32)卷积权重
                 out_height = ceil(float(in_height - filter_height + 1) / float(strides[1])) (28-3+1) / 1 = 26，(28-3+1) / 2 = 13
                 out_width = ceil(float(in_width - filter_width + 1) / float(strides[2])) (28-3+1) / 1 = 26，(28-3+1) / 2 = 13
                 2、使用使用SAME方式,feature map的尺寸为     (3,3,1,32)卷积权重
                 out_height = ceil(float(in_height) / float(strides[1]))  28 / 1 = 28，28 / 2 = 14
                 out_width = ceil(float(in_width) / float(strides[2]))   28 / 1 = 28，28 / 2 = 14
            ceil:函数返回数字的上入整数
      # 池化：
            ksize:第1，第4参数都为1，中间两个参数为池化窗口的大小，如：[1,1,1,1],[1,2,2,1]
                  实验证明：对于实际的池化后的数据尺寸，ksize没有影响，只是计算的范围不同。            
            strides：第1，第4参数都为1，中间两个参数为池化窗口的步幅，如：[1,1,1,1],[1,2,2,1]
                  实验证明：对于实际的池化后的数据尺寸，strides产生影响，具体的计算方式和卷积中的strides相同。'''

    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, conv2d_strides, padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        # Max Pooling
        maxPool = tf.nn.max_pool(hidden, ksize=pool_ksize, strides=pool_strides, padding='SAME')
        
        conv = tf.nn.conv2d(maxPool, layer2_weights, conv2d_strides, padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        # Max Pooling
        maxPool = tf.nn.max_pool(hidden, ksize=pool_ksize, strides=pool_strides, padding='SAME')
        
        #全连接层
        shape = maxPool.get_shape().as_list()
        # print(shape,'================================')
        reshape = tf.reshape(maxPool,[shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

        #增加dropout层
        hidden_drop = tf.nn.dropout(hidden, keep_prob)

        return tf.matmul(hidden_drop, layer4_weights) + layer4_biases
    
    
    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
             tf.nn.softmax_cross_entropy_with_logits(
                     labels=tf_train_labels, logits=logits))
     
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
    
    
num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        #当加入dropout时训练和测试keep_prob不一样，测试要保持全连接
        feed_dict_train = {tf_train_dataset:batch_data, tf_train_labels:batch_labels, keep_prob: 0.5}
        feed_dict_test = {tf_train_dataset:batch_data, tf_train_labels:batch_labels, keep_prob: 1}
        
        _, now_loss, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict_train)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, now_loss))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(
                        feed_dict=feed_dict_test), valid_labels))
            print('Test accuracy: %.1f%%' % accuracy(
                    test_prediction.eval(
                        feed_dict=feed_dict_test), test_labels))
    
    
    
