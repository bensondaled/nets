##
import tensorflow as tf
import numpy as np
from chords import Chords

## Generate data generation object

ch = Chords(n=3000, sigma=1.)

##

# Parameters
learning_rate = 0.009 #0.009 for RMS
n_epochs = 200
batch_size = 50 # too large→faster but not updating weights frequently enough

# Network Parameters
n_input = ch.clip_size # n samples in clip
n_hidden_1 = 10 # 25 works
n_hidden_2 = 15 # 10 works
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

def network(x, out_idx=-1):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']),biases['h1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2']))
    output = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out']),biases['out']))
    # sigmoid seems to work better than softmax for output
    return [layer_1,layer_2,output][out_idx]

lyr2= network(X, out_idx=1)
y_predicted = network(X)
y_predicted_val = network(Xval)

cost = tf.reduce_mean(tf.pow(y_predicted - Y, 2))
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_predicted, Y))
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

## test data

test_x,test_y = ch.test_data,ch.test_labs
out_y = sess.run(y_predicted, feed_dict={X: test_x})

lab_real = np.argmax(test_y,axis=1)
lab_pred = np.argmax(out_y,axis=1)
print(np.mean(lab_real==lab_pred))

for uvl in sorted(np.unique(lab_real)):
    where = uvl == lab_real
    pl.plot(uvl, np.mean(lab_pred[where] == lab_real[where]), 'ko')

## visualizing the intermediate layers

layer3,layer2 = sess.run([y_predicted,lyr2], feed_dict={X: test_x})

l3maj = layer3[lab_real==0]
l3min = layer3[lab_real==1]
l3dim = layer3[lab_real==2]
pl.plot(l3maj.mean(axis=0))
pl.plot(l3min.mean(axis=0))
pl.plot(l3dim.mean(axis=0))
pl.title('The average outputs of the network for different classes')

pl.figure()
l2maj = layer2[lab_real==0]
l2maj_mean = l2maj.mean(axis=0)
l2min = layer2[lab_real==1]
l2min_mean = l2min.mean(axis=0)
l2dim = layer2[lab_real==2]
l2dim_mean = l2dim.mean(axis=0)
pl.plot(l2maj_mean)
pl.plot(l2min_mean)
pl.plot(l2dim_mean)
pl.title('The mean activations of the second-last layer for each class')

wout = weights['out'].eval()
bout = biases['out'].eval()

# manually evaluate the result of turning layer 2 into an output using the final weights
# note that for now I'm not even using the sigmoid function, it's just linear
pl.figure()
pl.plot( (np.sum(l2maj_mean[:,None] * wout, axis=0) + bout) )
pl.plot( (np.sum(l2min_mean[:,None] * wout, axis=0) + bout) )
pl.plot( (np.sum(l2dim_mean[:,None] * wout, axis=0) + bout) )
pl.title('The manual activation of mean class activations across the final layer')

##
