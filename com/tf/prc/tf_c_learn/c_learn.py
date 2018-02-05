# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def c_learn():
    # 定义一个特性列表features。
    # 这里仅仅使用了real-valued特性。还有其他丰富的特性功能
    features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

    # 一个评估者（estimator）是训练（fitting）与评估（inference）的开端。
    # 这里预定于了许多类型的训练评估方式，比如线性回归（linear regression）、
    # 逻辑回归（logistic regression）、线性分类（linear classification）和回归（regressors）
    # 这里的estimator提供了线性回归的功能
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

    # TensorFlow提供了许多帮助类来读取和设置数据集合
    # 这里使用了‘numpy_input_fn’。
    # 我们必须告诉方法我们许多多少批次的数据，以及每次批次的规模有多大。
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4,
                                                  num_epochs=1000)

    # ‘fit’方法通过指定steps的值来告知方法要训练多少次数据
    estimator.fit(input_fn=input_fn, steps=1000)

    # 最后我们评估我们的模型价值。在一个实例中，我们希望使用单独的验证和测试数据集来避免过度拟合。
    print(estimator.evaluate(input_fn=input_fn))


if __name__ == "__main__":
    c_learn()
