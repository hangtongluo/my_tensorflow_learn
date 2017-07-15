# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 23:19:31 2017

@author: Administrator
"""

import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print('load data finishing..')


# 构建Softmax 回归模型
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))

# 类别预测与损失函数
# y = tf.nn.softmax(tf.matmul(x,W) + b)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
y = tf.add(tf.matmul(x, W), b)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# 训练模型
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

'''如果你没有使用InteractiveSession，那么你需要在启动session之前构建整个计算图，然后启动该计算图。'''
sess = tf.InteractiveSession()   #构建一个交互的计算图
sess.run(tf.global_variables_initializer())    #初始化Variable

for i in range(1000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x:batch[0], y_:batch[1]})


# 模型评价
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))





















































