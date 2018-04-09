##

"""
Goal 1: teach network to walk back and forth across state space
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
https://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://karpathy.github.io/2015/05/21/rnn-effectiveness/

https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
https://medium.com/@erikhallstrm/tensorflow-rnn-api-2bb31821b185
https://medium.com/@erikhallstrm/using-the-tensorflow-lstm-api-3-7-5f2b97ca6b73

"""

## Imports

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

## Generate data

data = np.zeros([int(1e4),4])
for idx,row in enumerate(data[1:], 1):
    lastrow = data[idx-1]
    if not np.any(lastrow == 1):
        newspot = np.random.choice([0,1,2,3], p=[.5,.3,.1,.1])
        #row[np.random.randint(len(row))] = 1
        row[0] = 1
    elif lastrow[-1] == 1:
        row[:] = 0
    else:
        is1 = np.argwhere(lastrow==1).squeeze()
        row[is1+1] = 1

def get_batch(batch_size, timesteps):
    # chunks of timestep+1 so I can take the last timestep as the Y (desired output)
    data_i = int(np.random.choice(np.arange(len(data)-(batch_size*(timesteps+1)))))
    batch_x = data[data_i:data_i+batch_size*(timesteps+1)]
    batch_x = np.array([batch_x[i::timesteps+1] for i in range(timesteps+1)]) # timesteps x batches x dimensionality
    batch_y = batch_x[-1,:,:]
    batch_x = batch_x[:-1,:,:]
    # reshape to batchsize x timesteps x dimensionality
    batch_x = np.transpose(batch_x, [1,0,2])

    return batch_x, batch_y

## Parameters

# network structure
n_input = data.shape[1]
n_output = n_input
timesteps = 3
n_hidden = 20

# training params
learning_rate = int(1e-3)
training_steps = int(1e3)
batch_size = 50

## Build structures

X = tf.placeholder(tf.float32, [None, timesteps, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

weights =   {
                'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
            }
biases =    {
                'out': tf.Variable(tf.random_normal([n_output]))
            }

def RNN(x, weights, biases):

    # x should be: a list of length <timesteps>, each with shape [batch_size, n_input]
    # so transpose from batchsize x timesteps x dimensionality
    x = tf.unstack(x, axis=1)

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.5, reuse=tf.AUTO_REUSE)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return outputs, states

outputs, states = RNN(X, weights, biases)
logits = [tf.matmul(output, weights['out']) + biases['out'] for output in outputs]
prediction = [tf.nn.softmax(logit) for logit in logits]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits,Y)]
loss_fxn = tf.reduce_mean(losses)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_fxn)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

## Run

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for step in range(training_steps):
        
        # prepare batch
        batch_x,batch_y = get_batch(batch_size, timesteps)
        
        # train batch
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # display update
        l_,p_ = sess.run([logits, prediction], feed_dict={X:batch_x, Y:batch_y})
        #print('\n\n\n')
        #print(batch_y)
        #print(p_)
        loss, acc = sess.run([loss_fxn, accuracy], feed_dict={X:batch_x, Y:batch_y})
        print(loss)

    # accuracy on test set
    test_x,test_y = get_batch(batch_size, timesteps)
    print( sess.run(accuracy, feed_dict={X: test_x, Y: test_y}) )

##
