import tensorflow as tf
import numpy as np

W = tf.Variable([[1,2,2],[2,3,3]], dtype=tf.float32, name = 'Weights')
b = tf.Variable([[1,2,3]],dtype=tf.float32,name ='biases')
print("Initialize: W b ")
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path: ", save_path)
