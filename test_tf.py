import tensorflow as tf
import numpy as np
#
# def assign_func(b):
#     b[:,:] = 0
#     return b
#
# a = tf.placeholder(tf.float32, shape=[1, 2], name = "tensor_a")
# b = tf.placeholder(tf.float32, shape=[None, None], name = "tensor_b")
# assign_v = tf.py_func(assign_func, [b], tf.float32)
# c = tf.nn.softmax(assign_v)
# sess = tf.Session()
# array_a = np.array([[1., 2.]])
# array_b = np.array([[3., 4.],[5., 6.],[7., 8.]])
# feed_dict = {a: array_a, b: array_b}
# tile_a_value = sess.run(c, feed_dict = feed_dict)
# print(tile_a_value)

import tensorflow.contrib.slim as slim

# with tf.name_scope('name_scope_1'):
with tf.variable_scope('v'):
    # var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var1 = slim.fully_connected([[1.0,2.0]], 10)
    var2 = slim.fully_connected([[1.0,2.0]], 10)

# with tf.name_scope('name_scope_1'):
with tf.variable_scope('v', reuse=True) as scope:
    scope.reuse_variables()
    # var2 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var3 = slim.fully_connected([[1.0,2.0]], 10)
    var4 = slim.fully_connected([[1.0,2.0]], 10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var3.name, sess.run(var3))
    print(tf.global_variables())
    tf.summary.FileWriter('./logs/', sess.graph)