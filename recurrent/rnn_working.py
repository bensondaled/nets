"""
do not modify: proof of concept for first shot at an RNN
"""
## Imports

import numpy as np
import tensorflow as tf

## Parameters
timesteps = 5
batch_size = 30
n_classes = 10

## Data synthesis

data = np.tile(np.arange(10), 1000)
def get_batch(size):
    i = np.random.randint(0, len(data)-size, size=size)
    bx = np.array([data[ii:ii+timesteps] for ii in i])
    by = np.array([data[ii+1:ii+1+timesteps] for ii in i])
    # right now this returns a single batch of size `size` that is *1-dimensional*, and contains multiple timesteps. you could imagine having multi dimensions next
    return bx,by

## Variable inits

batch_x = tf.placeholder(tf.float32, [batch_size, timesteps])
batch_y = tf.placeholder(tf.int32, [batch_size, timesteps])
#batch_y_oh = tf.one_hot(batch_y, n_classes, dtype=tf.int32)

init_state = tf.placeholder(tf.float32, [batch_size, timesteps])

W = tf.Variable(np.random.rand(timesteps+1, timesteps), dtype=tf.float32)
b = tf.Variable(np.zeros((1,timesteps)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(timesteps, n_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, n_classes)), dtype=tf.float32)

x_series = tf.unstack(batch_x, axis=1)
y_series = tf.unstack(batch_y, axis=1)

current_state = init_state
states_series = []
for xi in x_series:
    xi = tf.reshape(xi, [batch_size, 1])
    xi_concat = tf.concat([xi, current_state], 1)

    next_state = tf.tanh(tf.matmul(xi_concat, W) + b)
    states_series.append(next_state)
    current_state = next_state

logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,y_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(total_loss)

init = tf.global_variables_initializer()

##
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(10000):
        bx,by = get_batch(batch_size)
        _current_state = np.zeros((batch_size, timesteps))

        _total_loss, _train_step, _current_state, _predictions_series = sess.run(
            [total_loss, train_step, current_state, predictions_series],
            feed_dict={
                batch_x:bx,
                batch_y:by,
                init_state:_current_state
            })

        print(_total_loss)
    
##
