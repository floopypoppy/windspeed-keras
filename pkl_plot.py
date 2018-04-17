#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
pkl file data plotting

"""
file_name = 'adam_30_400_200_32_0.2-0.2_henorm_0.2_3000_2'
pklfile = open('/Users/xliu/Dropbox/expout/'+file_name+'.pkl','rb')
data = pickle.load(pklfile)
pklfile.close()
trnerr = data['trnerr']
vderr = data['vderr']
trnout = data['trnout']
vdout = data['vdout']
tstout = data['tstout']
trntarg = data['trntarg']
vdtarg = data['vdtarg']
tsttarg = data['tsttarg']

'''subplots'''
total_epoch = 100
trnrg, trnhists, trnbonds, trnx_mean, trnx_std = outDist(trnout,trntarg)
vdrg, vdhists, vdbonds, vdx_mean, vdx_std = outDist(vdout,vdtarg)
tstrg, tsthists, tstbonds, tst_mean, tst_std = outDist(tstout,tsttarg)
fig = plt.figure()
gs = gridspec.GridSpec(4, 3)
subplots_adjust(hspace=0.9)
ax1 = plt.subplot(gs[0,:])
dummy = np.array(trnerr)
plt.loglog(dummy[:total_epoch])
dummy = np.array(vderr)
plt.loglog(dummy[:total_epoch])
plt.legend(['training set error', 'validation set error'],fontsize='x-small')
plt.xlabel('epoch')
plt.ylabel('MSE error')
plt.title('training errors')
plt.tick_params(labelsize='small')
ax2 = plt.subplot(gs[1,:2])
plt.plot(trnbonds.T,trnhists.T)
plt.title('distribution of training set output',fontsize=11)
plt.legend(['v=5','v=6','v=7','v=8','v=9','v=10'],fontsize='x-small',ncol=2)
plt.tick_params(labelsize='small')
nor_tar = np.arange(5,10)/15.0
ax3 = plt.subplot(gs[1,2])
plt.errorbar(nor_tar,trnx_mean,yerr=trnx_std,fmt='.',color='grey',elinewidth=1)
plt.axis([0.3, 0.7, 0.3, 0.7])
#plt.axis([0,1,0,1])
plt.gca().set_aspect('equal', adjustable='box') # square axis
plt.ylabel('train set output') 
plt.tick_params(labelsize='small')
ax4 = plt.subplot(gs[2,:2], sharex=ax2)
plt.plot(vdbonds.T,vdhists.T)
plt.title('distribution of validation set output',fontsize=11)
plt.legend(['v=5','v=6','v=7','v=8','v=9','v=10'],fontsize='x-small',ncol=2)
plt.tick_params(labelsize='small')
ax5 = plt.subplot(gs[2,2])
plt.errorbar(nor_tar,vdx_mean,yerr=vdx_std,fmt='.',color='grey',elinewidth=1)
plt.axis([0.3, 0.7, 0.3, 0.7])
#plt.axis([0,1,0,1])
plt.gca().set_aspect('equal', adjustable='box') # square axis
plt.ylabel('validation set output') 
plt.tick_params(labelsize='small')

ax4 = plt.subplot(gs[3,:2], sharex=ax2)
plt.plot(tstbonds.T,tsthists.T)
plt.title('distribution of test set output',fontsize=11)
plt.legend(['v=5','v=6','v=7','v=8','v=9','v=10'],fontsize='x-small',ncol=2)
plt.tick_params(labelsize='small')
ax5 = plt.subplot(gs[3,2])
plt.errorbar(nor_tar,tst_mean,yerr=tst_std,fmt='.',color='grey',elinewidth=1)
plt.axis([0.3, 0.7, 0.3, 0.7])
#plt.axis([0,1,0,1])
plt.gca().set_aspect('equal', adjustable='box') # square axis
plt.xlabel('normalised expected output')
plt.ylabel('test set output') 
plt.tick_params(labelsize='small')
    