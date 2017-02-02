##
"""
The goal is here to make a neural net that can predict the language of a word (or sentence): either English for French (for now).

The training data consists of a word list of thousands of words from each language.

The Words class converts these lists into datasets with train, test, and validate subsets.

Input to the network consists of 26 units, one per character of the alphabet, with values being the frequency of that character in the word. Note that for this to work, there must essentially be a systematic difference in the frequency of characters in english and french words (which I suspect is true but did not explicitly test by other means).

As it stands, I've achieved 80+% accuracy on this, using no sequence info about the characters, and just a 2-hidden-layer network. However, this seems to be the limit using this approach.

Next goal: improve using recurrence.
"""
##
import codecs, string
import tensorflow as tf

class Words():
    """Load word files of 2 languages and convert to nets dataset

    Source: http://www.gwicks.net/dictionaries.htm

    Parameters
    ----------
    filename1 : str
        path to file 1
    filename2 : str
        path to file 2
    """
    def __init__(self, filename1, filename2):

        self.filename1 = filename1
        self.filename2 = filename2

        # load datafiles
        load_params = dict(encoding='utf8', mode='r', errors='ignore')
        with codecs.open(filename1, **load_params) as f1, codecs.open(filename2, **load_params) as f2:
            self.raw1 = f1.readlines()
            self.raw2 = f2.readlines()

        # filter
        self.raw1,self.raw2 = map(self.filter_words, [self.raw1,self.raw2])

        np.random.shuffle(self.raw1)
        np.random.shuffle(self.raw2)

        small = np.min([len(self.raw1), len(self.raw2)])
        self.raw1 = self.raw1[:small]
        self.raw2 = self.raw2[:small]

        # format into data and labels
        self.labels = np.array([0]*len(self.raw1) + [1]*len(self.raw2))
        self.data = np.array(self.raw1 + self.raw2)
        self.n = len(self.data)

        # split into train, test, validate
        p_train, p_test, p_val = .8, .1, .1
        n_train, n_test, n_val = [int(len(self.data)*p) for p in [p_train, p_test, p_val]]
        idxs = np.arange(self.n)
        np.random.shuffle(idxs)
        self.idx_train = idxs[:n_train]
        self.idx_test = idxs[n_train:n_train+n_test]
        self.idx_val = idxs[n_train+n_test:]
        # perform the split
        self.data_train = self.data[self.idx_train]
        self.labels_train = self.labels[self.idx_train]
        self.data_test = self.data[self.idx_test]
        self.labels_test = self.labels[self.idx_test]
        self.data_val = self.data[self.idx_val]
        self.labels_val = self.labels[self.idx_val]

        # runtime variables
        self.i = 0 # index in data for batch calling

    def get_data_val(self):
        return words_to_vecs(self.data_val)
    def get_data_test(self):
        return words_to_vecs(self.data_test)

    def filter_words(self, words):
        new_words = []
        for w in words:
            w = w.strip().lower()
            w = w.replace("'",'')
            w = ''.join([c for c in w if c.isalpha()])
            if len(w) < 3:
                continue
            if all([wi==w[0] for wi in w]):
                continue

            new_words.append(w)
        return new_words

    def batch(self, n):
        """Get a batch of data for an iteration

        """
        # retrieve data and labels
        x = np.take(self.data_train, np.arange(self.i,self.i+n), axis=0, mode='wrap')
        y = np.take(self.labels_train, np.arange(self.i,self.i+n), axis=0, mode='wrap')
        
        # reshape y
        y_ = np.zeros([len(y),2])
        y_[range(len(y_)),y] = 1
        y = y_

        # convert x to array
        x = words_to_vecs(x)

        # increment counter
        self.i += n
        if self.i >= len(self.data_train):
            self.i -= len(self.data_train)

        return x,y

class Dummy():
    """A class that acts like the Words class, but produces totally trivial data instead: one language corresponds to a single word, and the other to a different word. Used for investigating hyperparameters.
    """
    def __init__(self, *args):
        self.data_train,self.labels_train = self.batch(50000)
        self.data_val,self.labels_val = self.batch(5000)
        self.labels_val = np.argmax(self.labels_val, axis=1)
        self.data_test,self.labels_test = self.batch(5000)
        self.labels_test = np.argmax(self.labels_test, axis=1)
    def get_data_val(self):
        return self.data_val
    def get_data_test(self):
        return self.data_test
    def batch(self, n):

        y = np.random.choice([0,1], size=n)

        x = np.zeros([len(y), 26])
        for i in range(len(y)):
            if y[i] == 0:
                x[i,[0,11,23,24]] = 1
            elif y[i] == 1:
                x[i,[20,25,3,9]] = 1

        y_ = np.zeros([len(y),2])
        y_[range(len(y_)),y] = 1
        y = y_

        return x,y


def words_to_vecs(words):
    """Given a list of words, return a list of vectors, where each vector is a 26-element vector with values indicating frequency of each letter
    """
    vec = np.zeros([len(words), 26])
    for idx,w in enumerate(words):
        for ch in w:
            chidx = string.ascii_lowercase.index(ch)
            vec[idx,chidx] += 1

    return vec

## Open data

words = Words('english.txt', 'french.txt')
#words = Dummy('english.txt', 'french.txt')

## Network preparation

# Parameters
learning_rate = 0.03
n_epochs = 3
batch_size = 3000
iters_per_epoch = len(words.data_train)//batch_size
opt = tf.train.AdamOptimizer

# Layer params
layer_ns = [26, 100, 80, 2]
activations = [tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid]
n_layers = len(layer_ns)

# Data variables
X = tf.placeholder(tf.float32, [None, layer_ns[0]]) # for input batch data
Y = tf.placeholder(tf.float32, [None, layer_ns[-1]]) # for input batch labels
Xval = tf.Variable(words.get_data_val(), dtype=tf.float32) # validation data
Xtest = tf.Variable(words.get_data_test(), dtype=tf.float32) # test data

# Setup weights and biases
weights, biases = [], []
for idx,n in enumerate(layer_ns[1:]):
    w = tf.Variable(tf.random_normal([layer_ns[idx], n]))
    b = tf.Variable(tf.random_normal([n]))
    weights.append(w)
    biases.append(b)

# Define network
def net(x):
    # x : input vector to network
    # returns: list of layers
    layers = []
    l = x # before first layer, input is the data itself
    for idx,(n,a,w,b) in enumerate(zip(layer_ns,activations,weights,biases)):
        l = a(tf.add(tf.matmul(l,w),b))
        layers.append(l)
    return layers

# Training variables & operations
y_predicted = net(X)[-1] # last layer is the output of the network
y_predicted_val = net(Xval)[-1]
y_predicted_test = net(Xtest)[-1]
cost = tf.reduce_mean(tf.pow(y_predicted - Y, 2))
optimizer = opt(learning_rate).minimize(cost)

## Run network
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

# Display vars
costs = np.zeros(300)
costs[:] = np.nan
val_acc = np.zeros(300) # validation accuracy
val_acc[:] = np.nan
fig,axs = pl.subplots(2,1, sharex=True)
cost_line, = axs[0].plot(costs, marker='o', color='steelblue')
val_line, = axs[1].plot(val_acc, marker='o', color='steelblue')
axs[0].set_xlim([0, len(costs)+1])
axs[0].set_ylabel('Cost')
axs[1].set_ylabel('Validation Set Accuracy')

for ep in range(n_epochs):
    for itr in range(iters_per_epoch):
        print('Epoch {}, batch {}'.format(ep, itr))

        # get a data batch
        batch_x,batch_y = words.batch(batch_size)

        # update
        _, c, val = sess.run([optimizer, cost, y_predicted_val], feed_dict={X: batch_x, Y: batch_y})
        val_pred = np.argmax(val, axis=1)
    
        # draw
        costs = np.roll(costs, -1)
        costs[-1] = c
        val_acc = np.roll(val_acc, -1)
        val_acc[-1] = np.mean(val_pred == words.labels_val)
        cost_line.set_ydata(costs)
        val_line.set_ydata(val_acc)
        axs[0].set_ylim(np.nanmin(costs), np.nanmax(costs))
        axs[1].set_ylim(np.nanmin(val_acc), np.nanmax(val_acc))
        pl.draw()
        pl.pause(0.1)

## Test accuracy on test dataset

test, = sess.run([y_predicted_test])
test = np.argmax(test, axis=1)
print(np.mean(test == words.labels_test))

## Manual testing
sent_fr = "Il etait une fois au milieu d une foret epaisse une petite maison ou habitait une jolie petite fille nommee Petit Chaperon Rouge Un jour ensoleille sa maman lappela dans la cuisine de leur petite maison Elle frappa a la porte de la maison Elle jeta un coup doeil par la fenetre Elle vit trois bols de porridge sur la table de la cuisine mais il ne semblait y avoir personne dans la maison Alors Boucles dor entra dans la maison Oh quelle netait pas sage cette petite fille"
sent_eng = "Down, down, down. Would the fall never come to an end I wonder how many miles Ive fallen by this time she said aloud. I must be getting somewhere near the centre of the earth. Let me see that would be four thousand miles down I think (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a very good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) yes, thats about the right distance but then I wonder what Latitude or Longitude Ive got to Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say"

sent = sent_fr

sent = ''.join([c for c in sent if c.isalpha() or c==' ']).lower().split(' ')
sent = words_to_vecs(sent).astype(np.float32)
pr = net(sent)[-1]
out = sess.run(pr)
print(np.mean(np.argmax(out, axis=1)))
print('(<.5 means english, >.5 means french)')

##
