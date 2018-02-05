#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def first_tf():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    node3 = tf.add(node1, node2)
    sess = tf.Session()

    print("node3: ", node3)
    print("sess.run(node3): ", sess.run(node3))

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    # print(sess.run(adder_node, {a: 3, b: 4.5}))
    # print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

    add_and_triple = adder_node * 3
    # print(sess.run(add_and_triple, {a: 3, b: 4.5}))

    w = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    liner_model = w * x + b

    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(liner_model, {x: [1, 2, 3, 4]}))

    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(liner_model - y)
    loss = tf.reduce_sum(squared_deltas)
    fix_w = tf.assign(w, [-1.])
    fix_b = tf.assign(b, [1.])
    sess.run([fix_w, fix_b])

    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    # set the optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print(sess.run([w, b]))


if __name__ == "__main__":
    first_tf()
