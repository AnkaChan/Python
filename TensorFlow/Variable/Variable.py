from __future__ import print_function
import tensorflow as tf

a = tf.Variable(0);
b = tf.Variable(1);

c = a + b;

init = tf.global_variables_initializer();

with tf.Session() as sess:
	sess.run(init)
	print(sess.run(c))