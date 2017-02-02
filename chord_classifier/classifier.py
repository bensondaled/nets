##
import tensorflow as tf
import numpy as np
from chords import Chords

## Generate data generation object

ch = Chords()

##

# Parameters
learning_rate = 0.001
n_epochs = 40
batch_size = 40 # too large→faster but not updating weights frequently enough
nsamples = 10000# bc i can generate as many as i want

# Network Parameters
n_input = ch.clip_size # n samples in clip
n_hidden_1 = 25
n_hidden_2 = 10
n_output = ch.n_classes

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

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

def network(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']),biases['h1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2']))
    output = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out']),biases['out']))
    # sigmoid seems to work better than softmax for output
    return output

y_predicted = network(X)

cost = tf.reduce_mean(tf.pow(y_predicted - Y, 2))
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

##

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(nsamples/batch_size)

costs = np.zeros(n_epochs)
costs[:] = np.nan
cost_line, = pl.plot(costs, marker='o', color='steelblue')
pl.xlim([0, n_epochs+1])

for epoch in range(n_epochs):
    print(epoch, end='.', flush=True)
    for i in range(total_batch):
        batch_xs,batch_ys = ch.batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
    costs[epoch] = c
    cost_line.set_ydata(costs)
    pl.ylim(np.nanmin(costs), np.nanmax(costs[:epoch+1]))
    pl.draw()
    pl.pause(0.1)

##

test_x,test_y = ch.batch(10000)
out_y = sess.run(y_predicted, feed_dict={X: test_x})

lab_real = np.argmax(test_y,axis=1)
lab_pred = np.argmax(out_y,axis=1)
print(np.mean(lab_real==lab_pred))

wrong = np.argwhere(lab_real!=lab_pred).squeeze()
pl.hist(lab_real[wrong])
##
