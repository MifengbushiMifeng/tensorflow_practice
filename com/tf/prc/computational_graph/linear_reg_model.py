# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def linear_regression_model():
    # parameter of model
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)

    # the input of the model
    x = tf.placeholder(tf.float32)
    # the output of the model
    linear_model = W * x + b

    # the input of the loss
    y = tf.placeholder(tf.float32)
    # the model of the loss
    loss = tf.reduce_sum(tf.square(linear_model - y))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # the training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    # define the loop for the train
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # assess the training result
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print(curr_W)
    print(curr_b)
    print(curr_loss)


if __name__ == "__main__":
    linear_regression_model()
