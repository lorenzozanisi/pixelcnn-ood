from make_blobs import make_blob
import numpy as np
from astropy.io import fits
import sqlcl
import matplotlib.pylab as plt
import pandas as pd
from multiprocessing import Pool
from functools import partial

df = pd.read_csv('../data/photometric_catalog_for_Sersic_blobs_SerOnly.dat' )           
#df = df[75000:]
#df = df.iloc[:10]
df['objid'] = df['objid'].map(lambda x: str(x))
#names_weird = np.loadtxt('ids_weirdos.txt', dtype=str)
#df_weird = pd.DataFrame.from_dict({'objid':names_weird})
#df = df.merge(df_weird, on='objid')
#df = df.drop('galcount', axis=1)
#df = df.drop('PA_BULGE', axis=1)
#df = df.drop('PA_DISK',axis=1)
#df['R_DISK'] = df['R_DISK']/np.sqrt(df['BA_DISK'])
#df['R_BULGE'] = df['R_BULGE']/np.sqrt(df['BA_BULGE'])
df = df[['m_bulge', 'r_bulge', 'n_bulge', 'ba_bulge', 'objid','z']]
df['r_bulge1'] = df['r_bulge']*np.sqrt(df['ba_bulge'])
df = df.drop('r_bulge', axis=1)
df = df[['m_bulge', 'r_bulge1', 'n_bulge', 'ba_bulge', 'objid','z']]

df = df.query('n_bulge<6.2')
print(df)
print('processing')

with Pool() as p:
    p.map(make_blob, df.values)
