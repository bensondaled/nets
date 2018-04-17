# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## 

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Activation
import numpy as np

##

model = Sequential()
model.add(LSTM(50, input_shape=(None, 1), return_sequences=True))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

model.fit(x_train, y_train, batch_size=512, nb_epoch=epochs, validation_split=0.1)

##
