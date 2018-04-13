#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
pkl file data plotting

"""
def covmap(ma,mb):
    '''
    compute the covariance-displacement graph between two input 2-D slope arrays
    
    Parameters
    ------
    ma, mb: 2-D slope map between which to compute the covariance-displacement graph
    
    outputs:
        covmtx: 
    '''    
    l = np.shape(ma)[0]
    covmtx = np.empty((2*l-1,2*l-1))
    for delta_x in range(-l+1,l):
        for delta_y in range(-l+1,l):
            if delta_x<=0 and delta_y<=0:
                slidea = ma[:l-abs(delta_x),:l-abs(delta_y)].flatten()
                slideb = mb[abs(delta_x):,abs(delta_y):].flatten()
            if delta_x<0 and delta_y>0:
                slidea = ma[:l-abs(delta_x),delta_y:].flatten()
                slideb = mb[abs(delta_x):,:l-delta_y].flatten()
            if delta_x>0 and delta_y<0:
                slidea = ma[delta_x:,:l-abs(delta_y)].flatten()
                slideb = mb[:l-delta_x,abs(delta_y):].flatten()
            if delta_x>=0 and delta_y>=0:
                slidea = ma[delta_x:,delta_y:].flatten()
                slideb = mb[:l-delta_x,:l-delta_y].flatten()
            if size(slidea)>1:
                temp = np.stack((slidea,slideb)) 
                covmtx[l-1+delta_x,l-1+delta_y] = cov(temp)[0,1]
            else :
                covmtx[l-1+delta_x,l-1+delta_y] = 0               
    return covmtx

file_name = 'adam_30_400_100_32_0.2(hid)_henorm_0.2_0'
pklfile = open('/users/xliu/dropbox/expout/hid/'+file_name+'.pkl','rb')
data = pickle.load(pklfile)
pklfile.close()
trnerr = data['trnerr']
vderr = data['vderr']
trnout = data['trnout']
vdout = data['vdout']
trntarg = data['trntarg']
vdtarg = data['vdtarg']

'''subplots'''
total_epoch = 100
trnrg, trnhists, trnbonds, trnx_mean, trnx_std = outDist(trnout,trntarg)
vdrg, vdhists, vdbonds, vdx_mean, vdx_std = outDist(vdout,vdtarg)
fig = plt.figure()
gs = gridspec.GridSpec(3, 3)
subplots_adjust(hspace=0.9)
ax1 = plt.subplot(gs[0,:])
dummy = np.array(trnerr)
plt.loglog(dummy[:total_epoch+1])
dummy = np.array(vderr)
plt.loglog(dummy[:total_epoch+1])
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
plt.xlabel('normalised expected output')
plt.ylabel('normalised training set output') 
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
plt.xlabel('normalised expected output')
plt.ylabel('normalised validation set output') 
plt.tick_params(labelsize='small')
    