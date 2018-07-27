import tensorflow as tf
import numpy as np

W = tf.Variable([[1,2,2],[2,3,3]], dtype=tf.float32, name = 'Weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name ='biases')
print("Load: W b ")

saver = tf.train.Saver()

with tf.Session() as sess:
    # 提取变量
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))

