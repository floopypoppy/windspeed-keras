#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
split all data into training and validation set 

"""
trn_split = 0.6
vd_split = 0.2
tst_split = 0.2
n_sample = 3000
n_frame = 100
n_spd = 5
n_trn = int(n_sample*trn_split)
n_vd = int(n_sample*vd_split)
n_tst = int(n_sample*tst_split)

h = fits.open('trnslos_20180416_all.fits')
trnslos = h[0].data
h.close()
width = 72*n_frame*n_sample
start = int(width*trn_split)
end = int(width*(1-tst_split))
trnslos = trnslos.reshape((n_spd, width))
tst = trnslos[:,end:]
tst = tst.reshape(-1, 72*n_frame)
tst_tar = np.arange(5,10).repeat(n_tst)/15.0
header = fits.Header()
header['TS_SPLIT'] = str(tst_split)
header['N_SAMPLE'] = str(n_sample)
fits.writeto("tstslos_20180417.fits",tst,header,overwrite=True)
fits.writeto("tsttar_20180417.fits",tst_tar,header,overwrite=True)

trn = trnslos[:,:start]
trn = trn.reshape(-1, 72*n_frame)
vd = trnslos[:,start:end]
vd = vd.reshape(-1, 72*n_frame)
trn_vd = np.vstack((trn,vd))
tar = np.hstack((np.arange(5,10).repeat(n_trn),np.arange(5,10).repeat(n_vd)))/15.0
header = fits.Header()
header['TRN_SPLIT'] = str(trn_split)
header['VD_SPLIT'] = str(vd_split)
header['N_SAMPLE'] = str(n_sample)
fits.writeto("trnslos_20180417.fits",trn_vd,header)
fits.writeto("trntar_20180417.fits",tar,header)