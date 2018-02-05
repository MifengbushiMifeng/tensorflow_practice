# !/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np


# define a feature array
def model(features, labels, mode):
    # create linear model and the default value
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b

    # graph of loss
    loss = tf.reduce_sum(tf.square(y - labels))

    # graph of fitting
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # the ModelFnOps will create an abstract model by the specific conditions.
    return contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)


estimator = contrib.learn.Estimator(model_fn=model)

# define data
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., - 3.])

input_fn = contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# fitting data
estimator.fit(input_fn=input_fn, steps=1000)
print(estimator.evaluate(input_fn=input_fn, steps=10))

