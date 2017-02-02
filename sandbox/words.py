##
import csv
##
with open('sentences.csv') as f:
    dr = csv.DictReader(f, delimiter='\t', fieldnames=['idx','lang','txt'])
    lines = [line for line in dr if line['lang'] in ['eng','fra']]
##
data = pd.DataFrame(lines)

def nchar(s,n):
    if len(s) == n:
        st = s
    elif len(s) > n:
        st = s[:n]
    elif len(s) < n:
        st = '{:{length}s}'.format(s, length=n)

    abc = 'abcdefghijklmnopqrstuvwxyz'
    st = [st.count(l) for l in abc]
    return st

class Data():
    def __init__(self, data, n=10):
        self.data = data
        self.eng = self.data[self.data.lang=='eng'].txt.values
        self.fra = self.data[self.data.lang=='fra'].txt.values

        self.eng = [nchar(i,n) for i in self.eng]
        self.fra = [nchar(i,n) for i in self.fra]
        
        self.train_eng = self.eng[:20000]
        self.train_fra = self.fra[:20000]

        self.test_eng = np.asarray(self.eng[40000:60000])
        self.test_fra = np.asarray(self.fra[40000:60000])
    def next_batch(self, n):
        xs = []
        ys = []
        for i in range(n):
            lang = np.random.choice([0,1])
            dat = [self.train_eng,self.train_fra][lang]
            randi = np.random.choice(np.arange(len(dat)))
            strng = dat[randi]
            xs.append(strng)
            y = [0,0]
            y[lang] = 1
            ys.append(y)
        return np.asarray(xs),np.asarray(ys).reshape([-1,2])
    def get_test(self):
        xs = np.concatenate([self.test_eng,self.test_fra])
        ys = np.array([[1,0] for i in range(len(self.test_eng))] + [[0,1] for i in range(len(self.test_fra))])
        return xs,ys

##
import tensorflow as tf

str_len = 26
n_langs = 2

d = Data(data, n=str_len)

x_ = tf.placeholder(tf.float32, [None, str_len])
y_ = tf.placeholder(tf.float32, [None, n_langs])

##

layer_params = [
                dict(name='input',  n=str_len),
                dict(name='hl1',    n=100,        act=tf.nn.sigmoid),
                dict(name='hl2',    n=250,        act=tf.nn.sigmoid),
                dict(name='output', n=n_langs, act=tf.nn.sigmoid),
               ]

weights = [ tf.Variable(tf.random_normal([l1['n'],l2['n']])) 
            for l1,l2 in zip(layer_params[:-1], layer_params[1:]) ]
biases  = [ tf.Variable(tf.random_normal([l['n']]))
            for l in layer_params[1:] ]

layer = x_
for w,b,lp in zip(weights,biases,layer_params[1:]):
    layer = tf.add(tf.matmul(layer, w), b)
    layer = lp['act'](layer)
output = layer

#cost = tf.reduce_mean(tf.square(tf.sub(y_, output)))
#cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output), reduction_indices=[1]))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y_))
trainer = tf.train.GradientDescentOptimizer(learning_rate=2).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
sess.run(init)

##

for i in range(500):
    print('.', end='', flush=True)
    if i%50==0: print()
    batch_xs, batch_ys = d.next_batch(100)
    sess.run(trainer, feed_dict={x_: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
testx,testy = d.get_test()
print("\nAccuracy:", accuracy.eval({x_: testx, y_: testy}))

##
