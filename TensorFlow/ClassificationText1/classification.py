from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#from tensorflow.examples.tutorials.mnist import input_data
#from tf.keras.datasets.mnist import input_data
#mnist = tf.keras.datasets.mnist

import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
	
	
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_in: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x_in: v_xs, y_in: v_ys})
    return result
	
x_in = tf.placeholder(tf.float32,[None, 28*28])
y_in = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(x_in, 28 * 28, 10, activation_function = tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_in * tf.log(prediction),
reduction_indices=[1])) # loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x_in: batch_xs, y_in: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))