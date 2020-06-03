import pandas as pd
import numpy as np

def load_datasets(orig=True):

    if orig:
        TNG50 = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG50_Rot_blobsLike_orig_0.03_0.055_new.csv')[['objid','likelihood','galsky','skysig','LLR']]
        TNG = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG_Rot_blobsLike_orig_0.03_0.055_new.csv')
        Illustris = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/Illustris_Rot_blobsLike_orig_0.03_0.055_new.csv')[['objid','likelihood','LLR']]
        
        TNG50_mag = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/df_TNG50_ordered_Mgt9.5_skysub_orig.csv',sep=' ')[['objid','mag','sersic_n_r','flag_r', 'sersic_rhalf_r',\
                                                                                                                    'StellarMasses_in_r30pkpc','SFR_MsunPerYrs_in_all_10Myrs']]
        TNG_mag = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/df_TNG_ordered_Mgt9.5_skysub_orig.csv',sep=' ')[['objid','mag']]
        Illustris_mag = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/df_ill_ordered_Mgt9.5_skysub_orig.csv',sep=' ')#[['objid','mag']]
        
        TNG50 = TNG50.merge(TNG50_mag, on='objid')
        TNG = TNG.merge(TNG_mag, on='objid')
        Illustris = Illustris.merge(Illustris_mag, on='objid')      
        
    else:
        TNG50 = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG50_Rot_blobsLike_orig_0.03_0.055_magmatch.csv')
        TNG = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG_Rot_blobsLike_orig_0.03_0.055_magmatch.csv')
        Illustris = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/Illustris_Rot_blobsLike_orig_0.03_0.055_magmatch.csv')
        
        TNG_mag = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/df_TNG_ordered_Mgt9.5_skysub_magmatch.csv',sep=' ')[['objid','mag']]
        TNG50_mag = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/df_TNG50_ordered_Mgt9.5_skysub_magmatch.csv',sep=' ')[['objid','mag']]
        TNG50 = TNG50.merge(TNG50_mag, on='objid')
        TNG = TNG.merge(TNG_mag, on='objid')      
        
    TNG50_mhalo_sfr = pd.read_csv('/scratch/lzanisi/pixel-cnn/TNG50_cutouts/TNG50_mhalo.csv')
    TNG50 = TNG50.merge(TNG50_mhalo_sfr, on='objid')
    TNG50 = TNG50.rename(columns={'galsky':'sky [nmaggie]','flag_r':'flag_sersic','mag_x':'mag','StellarMasses_in_r30pkpc':'Mstar'})

    Illustris = Illustris.drop(columns='SFR')
    Illustris_SFR = pd.read_csv('/scratch/lzanisi/pixel-cnn/Illustris1/Illustris1_cat_new.csv' )[['objid','SFR']]
    Illustris = Illustris.merge(Illustris_SFR, on='objid')
   # SDSS = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/cleaned_df_bis_0.03_0.055_blobsLike_skysub.dat',sep=' ')[32000:]#[['objid','galcount','','LCentSat','mag']]
   # SDSS_all = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/SDSS_Rot_blobsLike_0.03_0.055_all.csv')[['galcount','likelihood','LLR']]
    Illustris = Illustris.drop(columns='Mstar') # this was the previous definition of Illustris - all mass within the subhalo
    Illustris_Mstar = pd.read_csv('/scratch/lzanisi/pixel-cnn/Illustris1/Illustris_Mstar30pkpc.csv')
    Illustris_Mstar = Illustris_Mstar.rename(columns={'StellarMasses_in_r30pkpc':'Mstar'})
    Illustris = Illustris.merge(Illustris_Mstar, on='objid')
    #SDSS_mag = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/cleaned_df_bis_0.03_0.055_blobsLike_skysub_onlymag.dat', sep= ' ')
    
   # SDSS = SDSS.merge(SDSS_all, on='galcount')
    
   # print(SDSS.columns)
    #SDSS = SDSS.query('MhaloL>0').sample(frac=9196/len(SDSS_all))
    #SDSS = SDSS_L.merge(SDSS, on='galcount')
    SDSS = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/SDSS_Rot_blobsLike_0.03_0.055_new.csv')
    SDSS_SFR = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/SDSS_SFR.csv')
    SDSS = SDSS.merge(SDSS_SFR, on='galcount')
    # add physical properties to TNG
    #df_f = pd.read_csv('/scratch/lzanisi/pixel-cnn/TNG_cutouts/snap_95_unfiltered_bis.csv')[['Unnamed: 0','SFR']]
    #TNG = pd.merge(TNG, df_f, left_on='Illustris_ID_2_2', right_on='Unnamed: 0')
    mhalo = pd.read_csv('/scratch/lzanisi/pixel-cnn/TNG_cutouts/snap_95_mhalomean.csv')[['Unnamed: 0', 'ParentDM','M_BH','SFR']]
    TNG = pd.merge(TNG, mhalo, left_on='Illustris_ID_2_2', right_on='Unnamed: 0')
    centrals_jstar = pd.read_csv('/scratch/lzanisi/pixel-cnn/TNG_cutouts/TNG_jstar_s95_lzanisi.csv') #this contains only centrals
    TNG = pd.merge(TNG, centrals_jstar, left_on = 'Illustris_ID_2_2', right_on='Illustris_ID_2_1', how='outer' )
    TNG['LCentSat'] = TNG['sJ_star_1re'].apply(lambda x: 0 if x!=x else 1)
    TNG['SFR'] = TNG['SFR'].apply(lambda x: np.log10(x) if x==x else np.nan)
    TNG50['SFR'] = TNG50['SFR'].apply(lambda x: np.log10(x) if x==x else np.nan)

    #clean
  #  TNG = TNG.query('likelihood>3000 & sersic_n>0 & LLR>-50 & sersic_n<7  & LLR<500')
  #  TNG50 = TNG50.query('likelihood>3000 & sersic_n_r>0 & LLR>-50 & sersic_n_r<7  & LLR<500')

  #  Illustris = Illustris.query('likelihood>3000 & sersic_n>0 & LLR>-50  &  sersic_n<7 & LLR<500')
  #  SDSS = SDSS.query('likelihood>3000  & LLR>-50  & LLR<500 & GalSky_err>0 ')
    
    #rename
    SDSS['sky [nmaggie]'] = SDSS['GalSky'].apply(lambda x: 10**(-0.4*(x-22.5)  )*0.396**2)
    Illustris = Illustris.rename(columns={'galsky':'sky [nmaggie]','sersic_n':'n_bulge'})
    TNG = TNG.rename(columns={'sky':'sky [nmaggie]','sersic_n':'n_bulge','mag_x':'mag','LogMass30':'Mstar','ParentDM':'Mhalo'}) 
    TNG50 = TNG50.rename(columns={'galsky':'sky [nmaggie]','sersic_n_r':'n_bulge','mag_x':'mag','StellarMasses_in_r30pkpc':'Mstar'})
    SDSS = SDSS.rename(columns={'MsMendSerExp':'Mstar','MhaloL':'Mhalo'})
    
    
    TNG['$logR_e \ [arcsec]$'] = TNG['sersic_rhalf'].apply(lambda x: np.log10(0.396*x))
    TNG['Re'] = TNG['$logR_e \ [arcsec]$'].copy()
    TNG['$n_{ser}$'] = TNG['n_bulge'].copy()
    TNG['sSFR'] = TNG.apply(lambda row: row.SFR-row.Mstar,axis=1)
    TNG['sSFR'] = TNG['sSFR'].apply(lambda x: -12.5 if x<-12.5 or x!=x else x)
    
    
    TNG50['$logR_e \ [arcsec]$'] = TNG50['sersic_rhalf_r'].apply(lambda x: np.log10(0.396*x))
    TNG50['$n_{ser}$'] = TNG50['n_bulge'].copy()
    TNG50['sSFR'] = TNG50.apply(lambda row: row.SFR-row.Mstar,axis=1)
    TNG50['sSFR'] = TNG50['sSFR'].apply(lambda x: -12.5 if x<-12.5 or x!=x else x)
    TNG50['Mstar'] = TNG50['Mstar'].apply(lambda x: x +np.log10(0.68))
    TNG50['Re'] =  TNG50['$logR_e \ [arcsec]$'].copy()
    
    Illustris['$logR_e \ [arcsec]$'] = Illustris['sersic_rhalf'].apply(lambda x: np.log10(0.396*x))
    Illustris['$n_{ser}$'] = Illustris['n_bulge'].copy()
    Illustris['sSFR'] = Illustris.apply(lambda row: row.SFR-row.Mstar,axis=1)
    Illustris['sSFR'] = Illustris['sSFR'].apply(lambda x: -12.5 if x<-12.5 or x!=x else x)
    Illustris['Re'] = Illustris['$logR_e \ [arcsec]$'].copy()
    
    SDSS['$logR_e \ [arcsec]$'] = SDSS['r_bulge'].apply(np.log10)
    #SDSS = SDSS[SDSS['$logR_e \ [arcsec]$']>-1]
    SDSS['$n_{ser}$'] = SDSS['n_bulge'].copy()
    SDSS['sSFR'] = SDSS.apply(lambda row: row.SFR-row.Mstar,axis=1)
    SDSS['sSFR'] = SDSS['sSFR'].apply(lambda x: -12.5 if x<-12.5 or x!=x else x)
    SDSS['MediansSFR'] = SDSS.apply(lambda row: row.MEDIANSFR-row.Mstar,axis=1)
    SDSS['MediansSFR'] = SDSS['MediansSFR'].apply(lambda x: -12.5 if x<-12.5 or x!=x else x)
    SDSS['Mhalo'] = SDSS['Mhalo'].apply(lambda x: 0 if x==0 else x + np.log10(1.1))
    SDSS['Re'] = SDSS['$logR_e \ [arcsec]$'].copy()
    SDSS = SDSS.query('LLR<600')
    
    SDSS['LCentSat'] = SDSS['LCentSat'].replace(to_replace=2, value=0)
   # SDSS_large  = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/Catalog_SDSS_complete.dat',sep=' ')[['galcount','NewLCentSat','MhaloL']]
   # SDSS = SDSS.merge(SDSS_large, on='galcount')

    #SDSS = SDSS.query('Mstar>10')
    #TNG = TNG.query('Mstar>10')
    #TNG50 = TNG50.query('Mstar>10')
    #Illustris = Illustris.query('Mstar>10')
    
    return SDSS, TNG50, TNG, Illustris