# approach from http://neuralnetworksanddeeplearning.com/
##

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = mnist.train.images
y = mnist.train.labels
x_ = tf.placeholder(tf.float32, [None, x.shape[1]])
y_ = tf.placeholder(tf.float32, [None, y.shape[1]])

##

layer_params = [
                dict(name='input',  n=x.shape[1]),
                dict(name='hl1',    n=120,        act=tf.nn.sigmoid),
                dict(name='output', n=y.shape[1], act=tf.nn.sigmoid),
               ]

weights = [ tf.Variable(tf.random_normal([l1['n'],l2['n']])) 
            for l1,l2 in zip(layer_params[:-1], layer_params[1:]) ]
biases  = [ tf.Variable(tf.random_normal([l['n']]))
            for l in layer_params[1:] ]

layer = x_
for w,b,lp in zip(weights,biases,layer_params[1:]):
    layer = tf.add(tf.matmul(layer, w), b)
    layer = lp['act'](layer)
output = layer

#cost = tf.reduce_mean(tf.square(tf.sub(y_, output)))
#cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output), reduction_indices=[1]))
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, y_))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y_))
trainer = tf.train.GradientDescentOptimizer(learning_rate=50).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
sess.run(init)

##

for i in range(3000):
    print('.', end='', flush=True)
    if i%80==0: print()
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(trainer, feed_dict={x_: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("\nAccuracy:", accuracy.eval({x_: mnist.test.images, y_: mnist.test.labels}))

##
