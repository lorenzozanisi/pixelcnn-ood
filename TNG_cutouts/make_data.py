from astropy.io import fits
from astropy.table import Table
import numpy as np
import requests
from multiprocessing import Pool
import pandas as pd
import pickle as pkl


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
    path = '/scratch/lzanisi/pixel-cnn/TNG_cutouts/0.045_skysub'    
    hdu =fits.PrimaryHDU(f)
    hdu.writeto(path+'/'+name, clobber=True)
    id = int(x.split('_')[1])
    return f, id



def get_files():
 #   names = np.loadtxt(filename, dtype=str)
  #  l = lambda x: int(x.split('/')[-1].split('_')[1])
  #  Names = np.array(list(map(l, names)))
    
    
    df_TNG = Table(fits.open('TNG_cat_forLorenzo')[1].data).to_pandas()
    df_TNG['LogMass30'] = df_TNG['LogMass30'].apply(lambda x: x-np.log10(0.67))
    df_TNG = df_TNG.query('LogMass30>10')
    ids = df_TNG['Illustris_ID_2_1'].values
    l = lambda x: './0.045/broadband_{}_FullReal.fits_r_band_FullReal.fits'.format(x)
    names = np.array(list(map(l, ids)))
    with Pool() as pool:
        data_gen,ids = zip(*pool.map(get_images,names))
    ids_df = pd.DataFrame({'Illustris_ID_2_1':ids})
    df_TNG_save = ids_df.merge(df_TNG, on='Illustris_ID_2_1') # same order as ids!
    df_TNG_save = df_TNG_save.rename({'Illustris_ID_2_1':'objid'}, axis=1)
    df_TNG_save.to_csv('/scratch/lzanisi/pixel-cnn/data/df_TNG_ordered_Mgt10.csv',sep=' ', index=False)
    save = lambda i: '/scratch/lzanisi/pixel-cnn/TNG_cutouts/0.045_skysub/'+str(i)+'_r.fits'
    to_save = np.array(list(map(save,ids)))
    np.savetxt('/scratch/lzanisi/pixel-cnn/TNG_cutouts/filenames_TNG_Mgt10_skysub.txt',to_save, fmt='%s')
    return data_gen,ids
    

TNG,ids = get_files()

dic = {'data':TNG, 'objid':ids}
with open('/scratch/lzanisi/pixel-cnn/TNG_cutouts/0.045_skysub.pkl', 'wb') as f:
    pkl.dump(dic, f)
#np.savez('/scratch/lzanisi/pixel-cnn/TNG_cutouts/0.045_processed_asinh_Mgt10', TNG)
