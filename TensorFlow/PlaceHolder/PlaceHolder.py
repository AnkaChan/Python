from __future__ import print_function
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = input1 + input2;

sess = tf.Session()

print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))