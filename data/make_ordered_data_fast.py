
# This file  reads and cuts a pandas DF with original SDSS data. Then a for cicle loops through all the 
# indices of thenew DF and drops all the objects for which the cutout is weird. These objects are not
# going to be saved. The good objects will be processed and stored in a folder as .fits files. 
# These files will be listed in a corresponding .txt file.
# The images will also be stored along with their relative objid in a.npz array.

# This construction makes sure that the order in data is kept, although slower to run than a .map operation.
import pickle as pkl
import pandas as pd
import numpy as np
from astropy.io import fits
import os
from multiprocessing import Pool
import itertools


def shuffle(f,r_pix):
    r_pix = int(r_pix)
    if r_pix < 5:
        N = 3
    elif r_pix >= 5 and r_pix < 10:
        N = 2
    elif r_pix >= 10 and r_pix < 15:
        N = 1.3
    elif r_pix >= 15 and r_pix < 25:
        N = 1
    elif r_pix >= 25 and r_pix < 35:
        N = 0.7
    elif r_pix >= 35:
        N = 0.5
    cut = int(N*r_pix)
    flat = f[128-cut:128+cut, 128-cut:128+cut].flatten()
    np.random.shuffle(flat)
    shape = np.shape(f[128-cut:128+cut, 128-cut:128+cut])
    f[128-cut:128+cut, 128-cut:128+cut] = np.reshape(flat, shape)
    return f

def clean_df(ind,galcount,aa,kk,airmass,galsky,r_pix=None):
    try:
        string = '/scratch/lzanisi/pixel-cnn/SDSS_cutouts/0.02_0.08_Mstar_gt10_raw/'+str(ind)+'_r.fits'  #these files are only  good cutouts - they are the INPUT 
        
        # normalize by sky level
        f = (fits.open(string)[0].data-1000)   #mag
        galsky_nmaggie = 10**(-0.4*(galsky-22.5)  )*0.396**2 # nmaggie
        cpn = 53.907*10**(-0.4*(22.5+aa+kk*airmass)) # counts/nmaggie
        galsky_counts = galsky_nmaggie*cpn          
        f = f-galsky_counts # nmaggie, sky subtracted
        
        if r_pix is not None:
            f = shuffle(f, r_pix)
        f = np.arcsinh(f[64:192,64:192]/1000)
        f = np.clip(f, a_min=None,a_max=1)
        
        to_save = '/scratch/lzanisi/pixel-cnn//SDSS_cutouts/0.02_0.08_Mstar_gt10_asinh/'+str(ind)+'_r.fits'
        hdu = fits.PrimaryHDU(f)
        hdu.writeto(to_save,clobber=True)
        return np.array(f),galcount

    except:
        return np.zeros((128,128)), np.nan

    
if __name__ =='__main__':
    print('reading catalog')
    df_meert = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/df_cleaned_goodBlobs_SerOnly.dat', sep=' ') # use this one for double test with blobs
#    df_meert  = df_meert.query('z<0.08 & z>0.02 & MsMendSerExp>10') , make sure to use this one if training on full sample
    df_sky = pd.read_csv('DR7_FieldInfo.csv') 
    df = df_meert.merge(df_sky, on='objid') # keep the same ordering as original data
    
    ids = df['objid'].values
    galcounts = df['galcount'].values
    aa = df['aa'].values
    kk = df['kk'].values
    airmass = df['airmass'].values
    galsky = df['GalSky'].values
    
    shuffle_pixels = False
    if shuffle_pixels:
        print('shuffling...')
        df['r_tot'] = df['r_tot']/(df['ba_tot'].apply(np.sqrt))
        arcsec_per_pix = 0.396
        r_pix = df['r_tot']/arcsec_per_pix
        print('building iterator...')
        iterator = [(i,o,r) for i,o,r in zip(ids,galcounts, r_pix)]
    else:
        print('building iterator...')
        iterator = [(i,o,a,k,m,g) for i,o,a,k,m,g in zip(ids,galcounts,aa,kk,airmass,galsky)]
        

    print('running')
    with Pool() as pool:
        fig, galc = zip(*pool.starmap(clean_df,iterator))
            
    print('done running')
    
    print('saving figures')
    mask = np.array(np.logical_not(np.isnan(galc)))
    fig = np.array(fig)
    galc = np.array(galc)
    fig = fig[mask,:,:]
    galc = galc[mask]
        
    print('saving files')
    app = {'galcount': galc} 
    app = pd.DataFrame.from_dict(app, dtype=np.int64)
    merged = app.merge(df, on='galcount')
    merged.index = np.arange(0,len(merged))
    merged.to_csv('cleaned_df_bis_0.02_0.08_blobsLike.dat', sep= ' ', index=False)

    filename = '../SDSS_cutouts/filename_0.02_0.08_Mstar_gt10_asinh_blobsLike'
    ids = merged['objid'].values

    save = lambda i: '/scratch/lzanisi/pixel-cnn/SDSS_cutouts/0.02_0.08_Mstar_gt10_asinh/'+str(i)+'_r.fits'
    to_save = np.array(list(map(save,ids)))
    np.savetxt(filename+'.txt',to_save, fmt='%s')

    train_size= 67000
    dic = {'data':fig[train_size:], 'objid':ids[train_size:]}
    with open('../SDSS_cutouts/0.02_0.08_Mstar_gt10_asinh_test_blobsLike.pkl','wb') as f:
        pkl.dump(dic, f, protocol=4)
    dic = {'data':fig[:train_size], 'objid':ids[:train_size]}
    with open('../SDSS_cutouts/0.02_0.08_Mstar_gt10_asinh_train_blobsLike.pkl','wb') as f:
        pkl.dump(dic, f, protocol=4)
