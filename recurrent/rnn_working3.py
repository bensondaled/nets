"""
do not edit - working example of basic tf RNN implementation
"""

## Imports

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

## Parameters
timesteps = 5
batch_size = 80
n_classes = 10

## Data synthesis

data = np.tile(np.arange(10), 1000)
def get_batch(size):
    i = np.random.randint(0, len(data)-size, size=size)
    bx = np.array([data[ii:ii+timesteps] for ii in i])
    bx = bx[:,None,:] # batch samples x input dimensionality x timesteps
    by = np.array([data[ii+timesteps] for ii in i])
    return bx,by

## Variable inits

batch_x = tf.placeholder(tf.float32, [batch_size, 1, timesteps])
batch_y = tf.placeholder(tf.int64, [batch_size])

W2 = tf.Variable(np.random.rand(timesteps, n_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, n_classes)), dtype=tf.float32)

init_state = tf.placeholder(tf.float32, [batch_size, timesteps])

x_series = tf.unstack(batch_x, axis=2)

cell = rnn.BasicRNNCell(timesteps, reuse=tf.AUTO_REUSE)
states_series, current_state = rnn.static_rnn(cell, x_series, init_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
logit = logits_series[-1]
pred = tf.nn.softmax(logit)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=batch_y)
loss = tf.reduce_mean(loss)

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

correct = tf.equal(tf.argmax(pred, 1), batch_y)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

##
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(5000):
        bx,by = get_batch(batch_size)
        _current_state = np.zeros((batch_size, timesteps))

        _total_loss, _train_step, _current_state, = sess.run(
            [loss, train_step, current_state],
            feed_dict={
                batch_x:bx,
                batch_y:by,
                init_state:_current_state
            })

        #print(_total_loss)

    testx,testy = get_batch(batch_size)
    acc = sess.run(accuracy, feed_dict={batch_x:testx, batch_y:testy, init_state:_current_state})
    print()
    print(acc)
    
##
