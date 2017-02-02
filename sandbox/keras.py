import numpy as np
import cv2
import keras as kr
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Convolution3D
import h5py, pandas as pd

def t2i(t, ts):
    ts = np.asarray(ts)
    t = np.atleast_1d(np.asarray(t))
    return np.array([np.argmin(np.abs(i-ts)).astype(int) for i in t]).squeeze()

# input data
data_path = '/Users/ben/data/nets/data_compressed_09.h5'
mov_path = '/Users/ben/data/nets/20160516172648.h5'
behav_data = pd.HDFStore(data_path)
mov_data = h5py.File(mov_path, 'r')
mov_ts = mov_data['ts'][:,0]
sesh = pd.to_datetime(behav_data.trials.session.unique()[-1])
sync = behav_data['sessions/{}/sync'.format(sesh.strftime('%Y%m%d%H%M%S'))]
sync_sesh2cam = sync.cam-sync.session
sync_ar2cam = sync.cam-sync.ar
select_where = 'session={}'.format(repr(sesh))
bdata = behav_data.select('analogreader', where=select_where)
bdata.index += sync_ar2cam
trials = behav_data.trials
trials = trials[(trials.session==sesh) & (trials.outcome<2)]
trial_starts = np.asarray(trials.start)+sync_sesh2cam
bdata.lickl = bdata.lickl>4 
bdata.lickr = bdata.lickr>4
bdata.puffl = bdata.puffl>9.5
bdata.puffr = bdata.puffr>9.5
bdata[['puffl','puffr']] = bdata[['puffr','puffl']]
# by here, common time frame is in place

for t in range(20,len(trial_starts)+1):
    tlims = trial_starts[t:t+2]
    mti0,mti1 = t2i(tlims, mov_ts)
    movi = mov_data['mov'][mti0:mti1]
    tsi = mov_ts[mti0:mti1]

    bi = bdata[(bdata.index>tlims[0]) & (bdata.index<tlims[1])]
    ari = 0

    for fr,ti in zip(movi, tsi):
        clos = int(t2i(ti, bi.index))
        qu = bi.iloc[ari:clos+1]
        if np.any(qu.lickl):
            fr[-10:,-100:] = 255
        if np.any(qu.lickr):
            fr[-10:,:100] = 255
        if np.any(qu.puffl):
            fr[:10,-100:] = 255
        if np.any(qu.puffr):
            fr[:10,:100] = 255
        ari = clos
        cv2.imshow('a', fr)
        cv2.waitKey(16)
    #break
cv2.destroyAllWindows()

####

model = kr.models.Sequential()

conv_layer = Convolution3D(100, 6, 6, 6, input_shape=[len(chunks), 1, z, y, x]) 

model.add(conv_layer)
model.add(Activation("softmax"))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

x = np.random.random([1000,10])
y = (x.sum(axis=1)>5).astype(int)

model.fit(x, y)
