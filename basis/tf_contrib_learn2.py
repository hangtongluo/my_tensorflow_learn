# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 22:58:52 2017

@author: lht
"""

# Building Input Functions with tf.contrib.learn (用input_fn去构建管道输入)
'''This tutorial introduces you to creating input functions in tf.contrib.learn.
 You'll get an overview of how to construct an input_fn to preprocess 
 and feed data into your models. Then, you'll implement an input_fn that 
 feeds training, evaluation, and prediction data into a neural network 
 regressor for predicting median house values'''

'''当你需要去做很多特征工程，可以定义一个input_fn来处理，
这样可以制造一个管道使整体简化'''

import tensorflow as tf 
import itertools
import pandas as pd 

tf.logging.set_verbosity(tf.logging.INFO)

'''Define the column names for the data set in COLUMNS. 
To distinguish features from the label, also define FEATURES and LABEL'''


COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)


# Defining FeatureColumns and Creating the Regressor
'''create a list of FeatureColumns for the input data, 
which formally specify the set of features to use for training. 
Because all features in the housing data set contain continuous values, 
you can create their FeatureColumns using the tf.contrib.layers.
real_valued_column() function:'''

#连续数据real_valued_column
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]


regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10, 10],
                                          model_dir="/tmp/boston_model")

# Building the input_fn for regressor
'''
feature_cols
A dict containing key/value pairs that map feature column names 
to Tensors (or SparseTensors) containing the corresponding feature data.
labels
A Tensor containing your label (target) values: the values your model aims to predict.'''
def input_fn(data_set):
	feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
	labels = tf.constant(data_set[LABEL].values)
	return feature_cols, labels


# Training the Regressor
regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)


# Evaluating the Model
ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)


# Retrieve the loss from the ev results and print it to output:
loss_score = ev['loss']
print("Loss: {0:f}".format(loss_score))


# Making Predictions
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
# .predict() returns an iterator; convert to a list and print predictions
predictions = list(itertools.islice(y, 6))
print ("Predictions: {}".format(str(predictions)))
















