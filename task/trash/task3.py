"""
in progress

probably re-copy from task2
"""

##

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn

##

with np.load('trials.npz') as data_file:
    timing = data_file['timing']
    choice = data_file['choice']
    dur = data_file['dur']

timing = timing[dur==3.8]
choice = choice[dur==3.8]

def downsample_timing(timing, n):
    # reduce timing to a version with n bins
    return np.array([i.sum(axis=-1) for i in np.array_split(timing, n, axis=-1)]).transpose([1,2,0])  

timing = downsample_timing(timing, 6)

data_x = timing # samples x 2 x timebins
data_y = choice

# negative control:
#data_y = np.random.choice(data_y, size=len(data_y), replace=False)

x_train,x_test,y_train,y_test = train_test_split(data_x, data_y, test_size=0.2)

def get_batch(n, from_test=False):
    if from_test:
        i = np.random.choice(np.arange(len(x_test)), size=n, replace=False)
        bx = x_test[i]
        by = y_test[i]
    else:
        i = np.random.choice(np.arange(len(x_train)), size=n, replace=False)
        bx = x_train[i]
        by = y_train[i]
    return bx,by

## Parameters
input_dims = data_x.shape[1]
timesteps = data_x.shape[2]
batch_size = 300
test_batch_size = len(x_test)
n_classes = 2
n_layers = 3

## Variable inits

batch_x = tf.placeholder(tf.float32, [None, input_dims, timesteps])
batch_y = tf.placeholder(tf.int64, [None])

W2 = tf.Variable(np.random.rand(timesteps, n_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, n_classes)), dtype=tf.float32)

init_state = tf.placeholder(tf.float32, [n_layers, 2, None, timesteps])
init_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(hl[0],hl[1]) for hl in tf.unstack(init_state, axis=0)])

x_series = tf.unstack(batch_x, axis=2)

cell = rnn.BasicLSTMCell(timesteps, reuse=tf.AUTO_REUSE)
cell = rnn.MultiRNNCell([cell]*n_layers)
states_series, final_state = tf.nn.dynamic_rnn(cell, x_series, initial_state=init_state)

logit = tf.matmul(final_state.h, W2) + b2
pred = tf.nn.softmax(logit)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=batch_y)
loss = tf.reduce_mean(loss)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

correct = tf.equal(tf.argmax(pred, 1), batch_y)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

##
sess = tf.InteractiveSession()
sess.run(init)
    
for i in range(500):
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

testx,testy = get_batch(test_batch_size, from_test=True)
_current_cell_state = np.zeros((test_batch_size, timesteps))
_current_hidden_state = np.zeros((test_batch_size, timesteps))
acc = sess.run(accuracy, feed_dict={batch_x:testx, batch_y:testy, cell_state:_current_cell_state, hidden_state:_current_hidden_state})
print()
print(acc)
    
##

