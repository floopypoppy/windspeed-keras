#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
plot errorbar of training and validation errors for multi-run

"""

pklfile = open('/users/xliu/dropbox/expout/mse_vs_do-both.pkl','rb')
data = pickle.load(pklfile)
pklfile.close()

trn = {}
vd = {}
for k,v in data.items():
    trn_mse = np.array(v[0:9:2])
    vd_mse = np.array(v[1:10:2])
    trn[k] = [trn_mse.mean(),trn_mse.std()]
    vd[k] = [vd_mse.mean(),vd_mse.std()]

keys = sorted(trn.keys())

plt.errorbar(keys, [trn[key][0] for key in keys], yerr = [trn[key][1] for key in keys])
plt.errorbar(keys, [vd[key][0] for key in keys], yerr = [vd[key][1] for key in keys])
plt.legend(['training set','validation set'],fontsize='small')
plt.xlabel('dropout ratio (both)')
plt.ylabel('MSE')
plt.title('error bar of training and validation MSE vs dropout ratio (both)')