##
import tensorflow as tf
import numpy as np
from chords import Chords

## Generate data generation object

ch = Chords(sigma=2.)

##

# Parameters
learning_rate = 0.005
n_epochs = 200
batch_size = 50 # too large→faster but not updating weights frequently enough

# Network Parameters
n_input = ch.clip_size # n samples in clip
n_hidden_1 = 25
n_hidden_2 = 10
n_output = ch.n_classes

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])
Xval = tf.Variable(ch.val_data, dtype=tf.float32) # validation data
val_labs = np.argmax(ch.val_labs, axis=1)

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
y_predicted_val = network(Xval)

cost = tf.reduce_mean(tf.pow(y_predicted - Y, 2))
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

##

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

total_batch = len(ch.train_data)//batch_size

costs,vperf = np.zeros([2,n_epochs])
cperfs = np.zeros([n_epochs,ch.n_classes])
costs[:],vperf[:],cperfs[:] = np.nan,np.nan,np.nan
fig,axs = pl.subplots(2,1,sharex=True)
cost_line, = axs[0].plot(costs, marker='o', color='steelblue')
vperf_line, = axs[1].plot(vperf, marker='o', color='black')
plines = axs[1].plot(cperfs, linewidth=1)
axs[0].set_xlim([0, len(costs)+1])

for epoch in range(n_epochs):
    print(epoch, end='.', flush=True)
    for i in range(total_batch):
        batch_xs,batch_ys = ch.batch(batch_size)
        _, c, val_pred = sess.run([optimizer, cost, y_predicted_val], feed_dict={X: batch_xs, Y: batch_ys})

    val_pred = np.argmax(val_pred, axis=1)
    vperf = np.roll(vperf, -1)
    vperf[-1] = np.mean(val_pred == val_labs)
    vperf_line.set_ydata(vperf)

    cperfs = np.roll(cperfs, -1, axis=0)
    print(cperfs.shape)
    for idx,uvl in enumerate(sorted(np.unique(val_labs))):
        where = uvl == val_labs
        cperfs[-1,idx] = np.mean(val_pred[where] == val_labs[where])
        plines[idx].set_ydata(cperfs[:,idx])

    costs = np.roll(costs, -1)
    costs[-1] = c
    cost_line.set_ydata(costs)
    
    axs[0].set_ylim(np.nanmin(costs), np.nanmax(costs))
    axs[1].set_ylim(np.nanmin(cperfs), np.nanmax(cperfs))

    pl.draw()
    pl.pause(0.1)

##

test_x,test_y = ch.test_data,ch.test_labs
out_y = sess.run(y_predicted, feed_dict={X: test_x})

lab_real = np.argmax(test_y,axis=1)
lab_pred = np.argmax(out_y,axis=1)
print(np.mean(lab_real==lab_pred))

wrong = np.argwhere(lab_real!=lab_pred).squeeze()
pl.hist(lab_real[wrong])
##
