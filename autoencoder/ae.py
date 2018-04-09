
##

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##

x_tn = mnist.train.images
y_tn = mnist.train.labels
x_tt = mnist.test.images
y_tt = mnist.test.labels

N,dim = x_tn.shape

##

layer_0 = tf.placeholder(np.float64, [-1, dim])
layer_1 = tf.layers.dense(layer_0, dim, activation=tf.nn.relu)
layer_2 = tf.layers.dense(layer_1, dim)
y_pd = layer_2

##

loss = tf.losses.mean_squared_error(, y_pd)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

##

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

##
