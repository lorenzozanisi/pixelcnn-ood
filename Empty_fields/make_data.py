from astropy.io import fits
import numpy as np
from multiprocessing import Pool
from glob import glob 
import pandas as pd
import itertools
import os
import pickle as pkl

def run(x):

    data = fits.open(x,ignore_missing_end=True)
    h = data[0].header
    aa = h['ZERO']
    kk = h['EXTC']
    airmass = h['AIRM']
    sky = h['SKY'] #nmaggies
    cpn = 53.907*10**(-0.4*(22.5+aa+kk*airmass))
        
    img = data[0].data*cpn - sky*cpn
    img = np.round(img,1)
    img = np.arcsinh(img/1000)
    img = np.clip(img, a_min=None,a_max=1)
    hdu = fits.PrimaryHDU()
    hdu.data = np.array(img)
    name = x.split('/')[-1] #includes .fits
    hdu.writeto('skysub_fields/'+name, clobber=True)
    return np.array(img),name

    
names = np.loadtxt('empty_fields.txt',dtype=str)
with Pool() as pool:
    data,ids = zip(*pool.map(run,names))
    
dic = {'data':data,'objid':ids}
with open('empty_fields.pkl', 'wb') as f:
    pkl.dump(dic,f,protocol=4)
    
dic_ids = {'objid':ids}
df = pd.DataFrame.from_dict(dic_ids)
df.to_csv('/scratch/lzanisi/pixel-cnn/data/df_empty_fields.dat', sep=' ', index=False) # not really useful, only needed to use same processing routines
  
