# coding: utf-8
import pandas as pd
import numpy as np
from glob import glob

name_blobs = glob('NewBlobs_SerOnly_skysub_FullReal_Uncalibrated/*') #np.loadtxt('../sersic_blobs/filename_blobs.txt', dtype=str)
print(name_blobs[:10])
def make_ids_blobs(x):
    filename = x.split('/')[-1]
    objid = filename.split('_')[0].strip("'")
    try:    
        return int(objid)
    except:
        return 0
        
ids_blobs = list(map(make_ids_blobs, name_blobs))
df = pd.read_csv('../data/df_cleaned_SerOnly.dat',sep=' ' )
#chi2 = pd.read_csv('../data/chi2nu_SerExp.dat',sep=' ')
#df = df.merge(chi2,on='galcount')

app = {'objid':ids_blobs}
app =pd.DataFrame.from_dict(app)
df_blobs = app.merge(df, on='objid') # we want the same order as app!
df_blobs.to_csv('../data/df_cleaned_goodBlobs_SerOnly.dat', sep=' ',index=False)

s1 = set(df.galcount.values)
s2 = set(df_blobs.galcount.values)
c = list(s1-s2)
df.index = df.galcount
dffailed = df.loc[c,:]

dffailed.to_csv('../data/df_cleaned_failedBlobs_SerOnly.dat', index=False, sep= ' ')
