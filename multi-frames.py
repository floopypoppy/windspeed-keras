'''
vary n_frame
''' 

from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # CPU only

import numpy as np
import pickle
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import gridspec
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.activations import relu
from keras.optimizers import RMSprop, SGD, Adam, Nadam
# from sklearn.utils import shuffle
from keras import initializers
from keras.callbacks import TerminateOnNaN, ReduceLROnPlateau
# from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K

batch_size = 32
epochs = 100
validation_split = 0.1
deltan = 0
num_1st_nodes = 500

h = fits.open('/home/xuewen/Desktop/windspeed/trnslos_20180405.fits')
trnslos_all = h[0].data 
h.close()  

# for i in range(n_frame-1,deltan-1,-1):
#     norm_inp[:,i*72:(i+1)*72] -= norm_inp[:,(i-deltan)*72:(i-(deltan-1))*72]
# norm_inp = norm_inp[:,deltan*72:]

# mx = norm_inp.max()
# mn = norm_inp.min()
# norm_inp = (norm_inp-mn)/(mx-mn) # 0-1 normalisation

norm_tar = fits.open('/home/xuewen/Desktop/windspeed/normtar_20180405.fits')[0].data

terminator = TerminateOnNaN()
lrreducer = ReduceLROnPlateau(verbose=0,factor=0.2,min_lr=1e-7)

dic={}

for n_frame in range(40,110,10):
    norm_inp = trnslos_all[:,:n_frame*72].copy()
    mx = np.abs(norm_inp).max() 
    norm_inp /= mx 
    norm_inp *= 0.5 # [-0.5,0.5] normalisation 
    lst = []
    trn_samples = int(norm_inp.shape[0]*(1-validation_split))
    trndata = norm_inp[:trn_samples]
    trntarg = norm_tar[:trn_samples]
    vddata = norm_inp[trn_samples:]
    vdtarg = norm_tar[trn_samples:]
    for i in range(5):
        model = Sequential()
        model.add(Dense(num_1st_nodes, input_shape=(72*(n_frame-deltan),), activation='relu'))
        model.add(Dropout(0.2))
		# model.add(Dense(num_2nd_nodes, activation='relu'))
		# model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))

		# model.summary()

        model.compile(loss='mean_squared_error',
                      optimizer=Adam(),
                      metrics=['mse'])
        history = model.fit(norm_inp, norm_tar,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_split=0.1,
                        callbacks=[terminator,lrreducer])

        file_name = 'adam_'+str(n_frame)+'_500_100_32_dp_'+str(i)
        trnout = model.predict(trndata)
        vdout = model.predict(vddata)
        trnerr = history.history['loss']
        vderr = history.history['val_loss']
        resdic = {'trnout' : trnout,
                  'trntarg' : trntarg,
                  'vdout' : vdout, 
                  'vdtarg' : vdtarg,
                  'trnerr' : trnerr,
                  'vderr' : vderr}
        outfile = open('/home/xuewen/Dropbox/expout/'+file_name+'.pkl','wb')
        pickle.dump(resdic,outfile)
        outfile.close()
        model.save('/home/xuewen/Dropbox/expout/'+file_name+'.h5')
        lst.append(trnerr[-1])
        lst.append(vderr[-1])
    dic[n_frame] = lst

dicfile = open('/home/xuewen/Dropbox/expout/mse_vs_frames.pkl','wb')
pickle.dump(dic, dicfile)
dicfile.close()




