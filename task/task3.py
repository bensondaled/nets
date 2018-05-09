"""
working

https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
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

timing = downsample_timing(timing, 14)

data_x = timing # samples x 2 x timebins
data_x = data_x.transpose([0,2,1])
data_y = choice
data_y_onehot = np.zeros([len(data_y),2])
data_y_onehot[data_y==0,0] = 1
data_y_onehot[data_y==1,1] = 1
data_y = data_y_onehot

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

    # bx is currently nsamples x timesteps x network_input_dimensionality
    return bx,by

## Parameters
input_dims = data_x.shape[2]
timesteps = data_x.shape[1]
n_hidden = 500
batch_size = 500
test_batch_size = len(x_test)
n_classes = data_y.shape[1]

## Variable inits

final_weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
final_bias = tf.Variable(tf.random_normal([n_classes]))

X = tf.placeholder('float', [None, timesteps, input_dims])
Y = tf.placeholder('float', [None, n_classes])

X_in = tf.unstack(X, timesteps, 1)

# network achitecture
lstm_layer = rnn.BasicLSTMCell(n_hidden, forget_bias=1, reuse=tf.AUTO_REUSE)
outputs,*_ = rnn.static_rnn(lstm_layer, X_in, dtype='float32')

prediction = tf.matmul(outputs[-1], final_weights) + final_bias

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=Y))
opt=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()

##
sess = tf.InteractiveSession()
sess.run(init)
    
for i in range(250):
    bx,by = get_batch(batch_size)
    _current_cell_state = np.zeros((batch_size, timesteps))
    _current_hidden_state = np.zeros((batch_size, timesteps))

    _, acc, los = sess.run(
        [opt, accuracy, loss],
        feed_dict={
            X:bx,
            Y:by,
        })

    print(los)

testx,testy = get_batch(test_batch_size, from_test=True)
sess.run(accuracy, feed_dict={X:testx, Y:testy})
##
