"""
file frozen: do not edit

overall goal: trying to build intuition about the belief that certain trial timings give rise to better performance, but the space for that is just huge. so see if an RNN can predict choices using trial timing better than it can with just nR and nL, and if so, study its structure

V1: basic feedforward net using current nr, nl to predict choice

interim conclusions: a deep feedforward net can get about 74% accuracy in predicting choice using only nR and nL

next step: try giving timing of stimuli in a recurrent net
"""
##

import tensorflow as tf
from sklearn.model_selection import train_test_split

##

data_file = '/Users/ben/data/puffs/merged/20180411_pillow/data_20180411.h5'
with pd.HDFStore(data_file) as h:
    trials = h.trials

data_x = trials[['nR','nL']].values
data_y = trials['choice'].values
data_y_onehot = np.zeros([len(data_y), 2])
data_y_onehot[:,0] = (data_y==0).astype(int)
data_y_onehot[:,1] = (data_y==1).astype(int)

# controls:
# shuffle x so predictive information is lost - should give 0.5 accuracy
#data_x = data_x[np.random.choice(np.arange(len(data_x)), replace=False, size=len(data_x))]
# predict side of trial - just needs to compute a > function, should get 100%
#data_y = trials['choice'].values

x_train,x_test, y_train,y_test = train_test_split(data_x, data_y, test_size=0.2)

##

# Parameters
learning_rate = 0.5
n_epochs = 50
batch_size = 500

# Network Parameters
n_input = data_x.shape[1]
n_hidden_1 = 25
n_hidden_2 = 25
n_output = data_y_onehot.shape[1]

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.int32, [None])

weights = {
'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
'out': tf.Variable(tf.random_normal([n_hidden_2, n_output])),
}
biases = {
'h1': tf.Variable(tf.random_normal([n_hidden_1])),
'h2': tf.Variable(tf.random_normal([n_hidden_2])),
'out': tf.Variable(tf.random_normal([n_output])),
}

def network(x, out_idx=-1):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']),biases['h1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2']))
    output = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out']),biases['out']))
    return [layer_1,layer_2,output][out_idx]

y_ = network(X)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=y_)
loss = tf.reduce_mean(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

##

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

total_batch = len(x_train)//batch_size

for epoch in range(n_epochs):
    #print(epoch, end='.', flush=True)
    for i in range(total_batch):
        batch_x = x_train[i*batch_size:(i+1)*batch_size]
        batch_y = y_train[i*batch_size:(i+1)*batch_size]
        _, l, _y = sess.run([optimizer, loss, y_], feed_dict={X: batch_x, Y: batch_y})

    print(l)


out_y = sess.run(y_, feed_dict={X: x_test})

lab_real = y_test
lab_pred = np.argmax(out_y,axis=1)
print(np.mean(lab_real==lab_pred))

##
