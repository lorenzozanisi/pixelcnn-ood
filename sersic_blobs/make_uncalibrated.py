from astropy.io import fits
import numpy as np
from multiprocessing import Pool
from glob import glob 
import pandas as pd
import itertools
import os
import pickle as pkl

def run(x,g):

    try:
        data = fits.open('NewBlobs_SerOnly_skysub_FullReal/NewBlobs/'+str(x)+'_r_FullReal.fits',ignore_missing_end=True)
        h = data[0].header
        aa = h['ZERO']
        kk = h['EXTC']
        airmass = h['AIRM']
        sky = h['SKY'] #nmaggies
        cpn = 53.907*10**(-0.4*(22.5+aa+kk*airmass))
        
        img = data[0].data*cpn - sky*cpn
        if img.shape !=(128,128):
            return np.zeros((128,128)), 0
        img = np.round(img,1)
        img = np.arcsinh(img/1000)
        img = np.clip(img, a_min=None,a_max=1)
        hdu = fits.PrimaryHDU()
        hdu.data = np.array(img)
        hdu.writeto('./NewBlobs_SerOnly_skysub_FullReal_Uncalibrated/'+str(x)+'_r.fits', clobber=True)
        return np.array(img), g
    except:
        return np.zeros((128,128)), 0
        
#load data and bujild iterator
df_SDSS = pd.read_csv('../data/cleaned_df_bis_0.02_0.08.dat', sep=' ')
iterator = [(i,o) for i,o in zip(df_SDSS['objid'].values, df_SDSS['galcount'].values)]

print('running')
with Pool() as pool:
    fig, galc = zip(*pool.starmap(run,iterator))


#masking zeros - unprocessed files
data = np.array(fig)
galc = np.array(galc)
mask = np.logical_not(np.ma.masked_equal(galc, 0).mask)
galc_ok = galc[mask]
data1 = np.asarray(data)[mask]

print('saving file names')
app = {'galcount': galc_ok} 
app = pd.DataFrame.from_dict(app, dtype=np.int64)
merged = app.merge(df_SDSS, on='galcount')
merged.to_csv('../data/df_cleaned_newblobs_SerOnly_skysub.dat', sep=' ',index=False)

filename = 'filenames_blobs_skysub'
ids = merged['objid'].values.astype(np.uint64)
save = lambda x: '/scratch/lzanisi/pixel-cnn/sersic_blobs/NewBlobs_SerOnly_skysub_FullReal_Uncalibrated/'+str(x)+'_r.fits'
to_save = np.array(list(map(save,ids)))
np.savetxt(filename+'.txt',to_save, fmt='%s')

print('saving pickled files')
train_size = 67000
train_set = data1[:train_size]
ids_train = ids[:train_size]
dic_train = {'data':train_set, 'objid':ids_train}

test_set = data1[train_size:]
ids_test = ids[train_size:]
dic_test = {'data':test_set, 'objid':ids_test}

with open('sersic_newblobs_SerOnly_skysub_FullReal_Uncalibrated_test.pkl','wb') as f:
    pkl.dump(dic_test,f, protocol=4)
with open('sersic_newblobs_SerOnly_skysub_FullReal_Uncalibrated_train.pkl','wb') as f:
    pkl.dump(dic_train,f, protocol=4)
        
    
