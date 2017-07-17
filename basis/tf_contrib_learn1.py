# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 22:58:52 2017

@author: lht
"""

'''tf.contrib.learn是TensorFlow的一个使得训练变得方便的高层api'''
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import tensorflow as tf 
import os 
import urllib
import numpy as np 

#data set 
iris_traindata = "iris_training.csv"
iris_trainurl = "http://download.tensorflow.org/data/iris_training.csv"

iris_testdata = "iris_test.csv"
iris_testurl = "http://download.tensorflow.org/data/iris_test.csv"

def main():
	# If the training and test sets aren't stored locally, download them.
	if not os.path.exists(iris_traindata):
		raw = urllib.request.urlopen(iris_trainurl).read()
		with open(iris_traindata, "wb+") as f:
			f.write(raw)

	if not os.path.exists(iris_testdata):
		raw = urllib.request.urlopen(iris_testurl).read()
		with open(iris_testdata, "wb+") as f:
			f.write(raw)

	# Load datasets.
	train_set = tf.contrib.learn.datasets.base.load_csv_with_header(
     	filename=iris_traindata,
      	target_dtype=np.int,
      	features_dtype=np.float32)
	# print(train_set)
	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
		filename=iris_testdata,
		target_dtype=np.int,
		features_dtype=np.float32)

	# Specify that all features have real-value data
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
	print("feature_columns", feature_columns)

  	# Build 3 layer DNN with 10, 20, 10 units respectively.
	classifier = tf.contrib.learn.DNNClassifier(
  		feature_columns= feature_columns,
  		hidden_units=[10,20,10],
  		n_classes = 3,
  		model_dir="temp/iris_model")

	# Define the training inputs
	def get_train_inpyt():
		x = tf.constant(train_set.data)
		y = tf.constant(train_set.target)
		return x, y

	def get_test_input():
		x = tf.constant(test_set.data)
		y = tf.constant(test_set.target)
		return x, y

	#训练模型
	classifier.fit(input_fn=get_train_inpyt, steps=2000)
	
	#评价模型
	accuracy_scores = classifier.evaluate(input_fn=get_test_input, steps=1)["accuracy"]
	print("\nTest accuracy：{0:.4f}\n".format(accuracy_scores))

	# Classify two new flower samples.
	def new_samples():
		return np.array(
			[[6.4, 3.2, 4.5, 1.5],
       		 [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

	predictions = list(classifier.predict(input_fn=new_samples))
	print("New Samples, Class Predictions: {}\n".format(predictions))






if __name__ == '__main__':
	main()





























































