"""
in progress
"""

## Imports

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

## Parameters
timesteps = 20
batch_size = 40
n_classes = 10

## Data synthesis

data = np.tile(np.arange(5), 10000)
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

cell_state = tf.placeholder(tf.float32, [batch_size, timesteps])
hidden_state = tf.placeholder(tf.float32, [batch_size, timesteps])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

x_series = tf.unstack(batch_x, axis=2)

cell = rnn.BasicLSTMCell(timesteps, state_is_tuple=True, reuse=tf.AUTO_REUSE)
states_series, final_state = rnn.static_rnn(cell, x_series, init_state)

logit = tf.matmul(final_state.h, W2) + b2
pred = tf.nn.softmax(logit)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=batch_y)
loss = tf.reduce_mean(loss)

train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

correct = tf.equal(tf.argmax(pred, 1), batch_y)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

##
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(1000):
        bx,by = get_batch(batch_size)
        _current_cell_state = np.zeros((batch_size, timesteps))
        _current_hidden_state = np.zeros((batch_size, timesteps))

        _total_loss, _train_step, = sess.run(
            [loss, train_step],
            feed_dict={
                batch_x:bx,
                batch_y:by,
                cell_state:_current_cell_state,
                hidden_state:_current_hidden_state,
            })

        print(_total_loss)

    testx,testy = get_batch(batch_size)
    _current_cell_state = np.zeros((batch_size, timesteps))
    _current_hidden_state = np.zeros((batch_size, timesteps))
    acc = sess.run(accuracy, feed_dict={batch_x:testx, batch_y:testy, cell_state:_current_cell_state, hidden_state:_current_hidden_state})
    print()
    print(acc)
    
##

##
