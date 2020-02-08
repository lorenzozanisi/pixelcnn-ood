from astropy.io import fits
from astropy.table import Table
import numpy as np
from scipy.ndimage import zoom
from multiprocessing import Pool
import pandas as pd
import pickle as pkl
import h5py


def get_images(x):
    f = fits.getdata(x)
    h = fits.getheader(x)
    aa = h['ZERO']
    airmass = h['AIRM']
    kk = h['EXTC']
    galsky = h['sky'] #nmaggie
    cpn = 53.907*10**(-0.4*(22.5+aa+kk*airmass)) # counts/nmaggie
    galsky_counts = galsky*cpn          
    f = f-galsky_counts    
    f = np.arcsinh(f/1000)
    f = np.clip(f,a_min=None, a_max=1)
        
    name = x.split('/')[-1]
    path = '/scratch/lzanisi/pixel-cnn/Illustris1/0.045_skysub'    
    hdu =fits.PrimaryHDU(f)
    hdu.writeto(path+'/'+name, clobber=True)
    id = int(x.split('_')[1])
    return f, id

def get_files():
    df_ill = pd.read_csv('Illustris_cat_morpho.csv')
    df_ill = df_ill.query('Mstar>10')
    ids = df_ill['objid'].values
    l = lambda x: '/scratch/mhuertas/Illustris1/Outputs/Outputs/broadband_{}_FullReal.fits_r_band_FullReal.fits'.format(x)
    names = np.array(list(map(l, ids)))
    with Pool() as pool:
        data_gen,ids = zip(*pool.map(get_images,names))
    ids_df = pd.DataFrame({'objid':ids})
    df_ill_save = ids_df.merge(df_ill, on='objid') # same order as ids!
    df_ill_save.to_csv('/scratch/lzanisi/pixel-cnn/data/df_ill_ordered_Mgt10.csv',sep=' ', index=False)
    save = lambda i: '/scratch/lzanisi/pixel-cnn/Illustris1/0.045_skysub/'+str(i)+'_r.fits'
    to_save = np.array(list(map(save,ids)))
    np.savetxt('/scratch/lzanisi/pixel-cnn/Illustris1/filenames_ill_Mgt10_skysub.txt',to_save, fmt='%s')
    return data_gen,ids

figs,ids = get_files()

dic = {'data':figs, 'objid':ids}
with open('/scratch/lzanisi/pixel-cnn/Illustris1/0.045_skysub.pkl', 'wb') as f:
    pkl.dump(dic, f)