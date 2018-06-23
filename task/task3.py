"""
working - do not edit
a better LSTM implementation that properly allocates hidden units

https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
https://stackoverflow.com/questions/37901047/what-is-num-units-in-tensorflow-basiclstmcell
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

if you want to try keras:
    http://adventuresinmachinelearning.com/keras-lstm-tutorial/
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

#timing = downsample_timing(timing, 10)

data_x = timing # samples x 2 x timebins
data_x = data_x.transpose([0,2,1])
data_y = choice
# positive control:
#data_y = np.argmax(data_x.sum(axis=1), axis=-1)

data_y_onehot = np.zeros([len(data_y),2])
data_y_onehot[data_y==0,0] = 1
data_y_onehot[data_y==1,1] = 1
data_y = data_y_onehot

# negative control:
#data_y = data_y[np.random.randint(len(data_y), size=len(data_y))]

# r-l for trials
rl = np.diff(data_x.sum(axis=1), axis=-1).squeeze()
# correct side
side = np.argmax(data_x.sum(axis=1), axis=-1)
# choice of subject
choice = np.argmax(data_y, axis=1)
# subject got it correct
correct = side==choice

# use only error trials:
#data_x = data_x[~correct]
#data_y = data_y[~correct]

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
n_hidden = 16 # can work with as low as 3 hidden units
batch_size = 200
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
opt=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()

##
sess = tf.InteractiveSession()
sess.run(init)
    
for i in range(1000):
    bx,by = get_batch(batch_size)
    _current_cell_state = np.zeros((batch_size, timesteps))
    _current_hidden_state = np.zeros((batch_size, timesteps))

    _, los = sess.run(
        [opt, loss],
        feed_dict={
            X:bx,
            Y:by,
        })

    if i%10==0:
        print(f'{i} : {los}')

sess.run(accuracy, feed_dict={X:x_test, Y:y_test})

## inspections

# exploring trained performance on the entire dataset

# predictions of the rnn on each trial
pred = sess.run(prediction, feed_dict={X:data_x, Y:data_y})
# binarized to left or right
pred_bin = np.argmax(pred, axis=1)
# confidence proxy for prediction?
margin = np.abs(np.diff(pred, axis=1)).squeeze()

# rnn matches subject choices or correct side
match_ch = choice==pred_bin
match_si = side==pred_bin # aka rnn got trial "correct"

res = pd.DataFrame(np.array([rl, margin, pred_bin, choice, side, match_ch, match_si, correct]).T, 
        columns=['rl','margin','net_pred','choice','side','match_ch','match_si','correct'])

# rnn learns how to predict *correct* decisions, but not incorrect ones
# in other words, rnn learns how to perform the task, but doesn't capture the lapses (i.e. it gets a trial correct when the subjects fail on it)
# so the rnn has learned the task, but has not learned how to accurately simulate an animal, particularly in its lapses (if it captured lapses, 1st value here would be high)
# and the fact that the match on error trials is *so* low, not near 50, suggests it totally learns the task in favour of learning the lapse strategy
res.groupby('correct').match_ch.mean()

# rnn predicts animal's choices on easy trials better than on hard trials
# as to be expected, since lapses, which it fails to capture, are more common in hard trials
res.groupby(pd.cut(res.rl.abs(), 6)).match_ch.mean().plot()

# rnn is less "confident" in trials where animals lapsed
# but that's just an indicator that the rnn is accumulating, and thus has a stronger decision variable in easier trials
res.groupby('correct').margin.mean()

# so this suggests I should train on error trials only - except of course that is another trivial problem because it then just has to learn the inverse rule
# but it may be possible to have a net "fail" in the same way as the subjects, and confirm it by inspecting psychometrics, etc.

##
