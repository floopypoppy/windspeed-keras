#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
auxiliaries of trial.py to evaluate network

"""
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.pyplot import imshow
from pylab import *
import pickle    

def outDist(output, target): 
    '''
    calculate distribution of actual output (all grouped according to different targets)
    
    Parameters
    ------
    output : network actual outputs
    target : corresponging targets
    
    outputs:
        rg: range of output
        bonds, hists: x and y axis of histogram of actual outputs
        x_mean, x_std: mean of output (grouped)
    '''
    counts = [list(target).count(i/15.0) for i in range(5,10)]   
    output = np.array(output)
    targer = np.array(target)
    rg = (output.min(),output.max())
    bs = 20
    hists = np.empty((5,bs))
    bonds = np.empty((1,bs))
    x_mean = np.empty(5)
    x_std = np.empty(5)
    start = 0
    i = 0
    for stride in counts:
        end = start + stride
        data = output[start:end]
        x_mean[i] = np.mean(data)
        x_std[i] = np.std(data)
        hists[i],temp,_ = plt.hist(data,bins=bs,range=rg)
        plt.close()
        start = end
        i += 1
    bonds[0] = [(temp[j]+temp[j+1])/2 for j in range(bs)]
    return rg, hists, bonds, x_mean, x_std

'''training data augmentation'''
n_frame = 100
trnslos = np.zeros((5000,72*n_frame))
for i in range(5,10):
    h = fits.open('trnslos_20180414_%i.fits'%i)
    trnslos[(i-5)*1000:(i-4)*1000] = h[0].data
    h.close()

'''open training slopes'''
h = fits.open('trnslos_20180405.fits')
trnslos_all = h[0].data
h.close()

'''remove mean from raw x and y slopes'''
trnslos = trnslos.reshape((5000*100*2,36))
nor_trnslos = np.empty((1000000,36))
for i in range(1000000):
    tmp = trnslos[i].copy()
    nor_trnslos[i] = tmp-tmp.mean()
nor_trnslos = nor_trnslos.reshape((5000,7200))
header = fits.Header()
header["r_0"] = str([0.16])
header["WINDSPD"] = str([5,6,7,8,9])
header["WINDDIR"] = str([0])
header["SAVETIME"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
header["ITERS"] = str([1000])
header['ITERTIME'] = str([0.012])
fits.writeto('trnslos_20180416.fits',nor_trnslos)

'''multi sets augmentation'''
n_frame = 100
trnslos = np.zeros((15000,72*n_frame))
h = fits.open('trnslos_20180414.fits')
tmp = h[0].data
h.close()
for i in range(0,15,3):
    j = i/3
    trnslos[i*1000:(i+2)*1000] = tmp[j*2000:(j+1)*2000]
h = fits.open('trnslos_20180416.fits')
tmp = h[0].data
h.close()
for k in range(2,17,3):
    j = (k-2)/3
    trnslos[k*1000:(k+1)*1000] = tmp[j*1000:(j+1)*1000]
header["r_0"] = str([0.16])
header["WINDSPD"] = str([5,6,7,8,9])
header["WINDDIR"] = str([0])
header["SAVETIME"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
header["ITERS"] = str([3000])
header['ITERTIME'] = str([0.012])
fits.writeto('trnslos_20180416_all.fits',trnslos)
