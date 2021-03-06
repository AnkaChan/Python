from __future__ import print_function
import tensorflow as tf
import numpy as np

x_data = np.random.rand(1000).astype(np.float32)
y_data = x_data*x_data*2.5 + x_data*3 + 3.3

Weights = tf.Variable(tf.random_uniform([2], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights[0] * x_data * x_data + Weights[1] * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
### create tensorflow structure end ###

sess = tf.Session()
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for step in range(10001):
    sess.run(train)
    if step % 100 == 0:
        print(step, sess.run(Weights), sess.run(biases))