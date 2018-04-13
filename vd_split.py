#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
split all data into training and validation set 

"""

vd_split = 0.2
n_sample = 1000
n_spd = 5
n_trn = int(n_sample*(1-vd_split))

h = fits.open('trnslos_20180327.fits')
trnslos = h[0].data
h.close()
trnslos = list(trnslos)

tmp = []
for i in range(n_spd):
    tmp.extend(trnslos[(i*n_sample):(i*n_sample+n_trn)])
for i in range(n_spd):
    tmp.extend(trnslos[(i*n_sample+n_trn):((i+1)*n_sample)])
tmp = np.array(tmp)

header = fits.Header()
header['VD_SPLIT'] = str(vd_split)
fits.writeto("trnslos_20180412.fits",tmp,header)

tar = np.empty([5000,1])
tar[:n_trn*n_spd,0] = np.arange(5,10).repeat(n_trn)
tar[n_trn*n_spd:,0] = np.arange(5,10).repeat(n_sample-n_trn)
tar /= 15.0
fits.writeto("normtar_20180412.fits",tar,header)