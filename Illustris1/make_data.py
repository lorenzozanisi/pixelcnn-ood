from astropy.io import fits
from astropy.table import Table
import numpy as np
from scipy.ndimage import zoom
from multiprocessing import Pool
import pandas as pd
import pickle as pkl
import h5py
import os


def get_images(x):
    
    name = x.split('/')[-1]
    id = int(name.split('_')[1])
    try:
        f_counts = fits.getdata(x)
        #f_nmaggies = fits.getdata(x) 
        h = fits.getheader(x)
        aa = h['ZERO']
        airmass = h['AIRM']
        kk = h['EXTC']
        galsky = h['sky'] #nmaggie
        skysig = h['skysig']
        cpn = 53.907*10**(-0.4*(22.5+aa+kk*airmass)) # counts/nmaggie
        galsky_counts = galsky*cpn          
        #f_counts = f_nmaggies*cpn
        f_nmaggies = f_counts/cpn - galsky
        mag = 22.5-2.5*np.log10(np.sum(f_nmaggies))
        
        f = f_counts-galsky_counts    
        #f = f_counts-galsky_counts
        f = np.arcsinh(f/1000)
        f = np.clip(f,a_min=None, a_max=1)
        name = x.split('/')[-1]
       # path = '/scratch/lzanisi/pixel-cnn/Illustris1/0.045_skysub_raw'    
       # hdu =fits.PrimaryHDU(f)
       # hdu.writeto(path+'/'+name, clobber=True)
        id = int(name.split('_')[1])
        F = f[32:96,32:96].flatten()
        num_ones = len(F[F==1])/len(F)
    except:
        os.system("echo {}>>failed.txt".format(id))
        return 0,0,0,0,0,0
    return f, id, galsky, skysig, mag,num_ones

def get_files():
    #df_ill = pd.read_csv('Illustris_cat_morpho.csv', sep= ' ')
    #df_ill = df_ill.query('Mstar>10')
    df_ill = pd.read_csv('Illustris1_cat.csv')
    #df_ill = df_ill.merge(df_sfr, on='objid')
    ids = df_ill['objid'].values
    l = lambda x: '/scratch/mhuertas/Illustris1/Outputs_orig/Outputs_orig/broadband_{}_FullReal.fits_r_band_FullReal.fits'.format(x)
    names = np.array(list(map(l, ids)))
    with Pool() as pool:
        data_gen,ids, galsky, skysig, mag, num_ones = zip(*pool.map(get_images,names))
    ids_df = pd.DataFrame({'objid':ids,'galsky':galsky,'skysig':skysig, 'mag':mag,'num_ones':num_ones})
    df_ill_save = ids_df.merge(df_ill, on='objid') # same order as ids!
    print(list(df_ill_save.columns))
    df_ill_save.to_csv('/scratch/lzanisi/pixel-cnn/data/df_ill_ordered_Mgt9.5_skysub_orig.csv',sep=' ', index=False)
    #save = lambda i: '/scratch/lzanisi/pixel-cnn/Illustris1/0.045_skysub_raw/'+str(i)+'_r.fits'
    #to_save = np.array(list(map(save,ids)))
    #np.savetxt('/scratch/lzanisi/pixel-cnn/Illustris1/filenames_ill_Mgt10_skysub_raw.txt',to_save, fmt='%s')
    return data_gen,ids

figs,ids = get_files()

dic = {'data':figs, 'objid':ids}

with open('/scratch/lzanisi/pixel-cnn/Illustris1/0.045_skysub_orig_Mgt9.5.pkl', 'wb') as f:
    pkl.dump(dic, f)
