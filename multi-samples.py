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

n_frame = 30
batch_size = 32
epochs = 200
validation_split = 0.2
deltan = 0
num_1st_nodes = 400

terminator = TerminateOnNaN()
lrreducer = ReduceLROnPlateau(verbose=0,factor=0.2,min_lr=1e-7)

dic={}

for sample in range(1000,4000,1000):
    h = fits.open('/home/xuewen/Desktop/windspeed/trnslos_'+str(sample)+'.fits')
    trnslos_all = h[0].data
    h.close()
    norm_tar = fits.open('/home/xuewen/Desktop/windspeed/normtar_'+str(sample)+'.fits')[0].data
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
        model.add(Dropout(0.2, input_shape=(72*(n_frame-deltan),)))
        model.add(Dense(num_1st_nodes, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mean_squared_error',
                      optimizer=Adam(),
                      metrics=['mse'])
        history = model.fit(norm_inp, norm_tar,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_split=validation_split,
                        callbacks=[terminator,lrreducer])

        file_name = 'adam_30_400_200_32_0.2-0.2_henorm_0.2_'+str(sample)+'_'+str(i)
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
    dic[sample] = lst

dicfile = open('/home/xuewen/Dropbox/expout/mse_vs_samples.pkl','wb')
pickle.dump(dic, dicfile)
dicfile.close()




