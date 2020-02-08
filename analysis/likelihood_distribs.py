import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from scipy.integrate import cumtrapz
import pandas as pd
mpl.rcParams['font.size']=45
#mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 3.
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['axes.linewidth']=5
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 20 
plt.rcParams['figure.figsize']=(16,12)


SDSS_orig = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/SDSS_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test.csv')
TNG_orig = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/TNG_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test.csv')
Illustris_orig = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/Illustris_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test.csv')
#noise_orig = pd.read_csv('noisetest_df_likelihood_asinh_Mgt10_3resnets_rightOrder_800ksteps.csv',
#                        delim_whitespace=True)
blobs_orig = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/blobs_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test.csv')
sky_orig = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/sky_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test.csv')
sky_orig_norot = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/sky_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test_noRot.csv')


SDSS_shuffled = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/SDSS_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test.csv')
TNG_shuffled = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/TNG_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test.csv')
Illustris_shuffled = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/Illustris_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test.csv')
#noise_shuffled = pd.read_csv('noisetest_df_likelihood_asinh_SersicBlobs_SerOnly_1Msteps.csv'#,
#                            delim_whitespace=True)
blobs_shuffled = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/blobs_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test.csv')
sky_shuffled = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/sky_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test.csv')
sky_shuffled_norot = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/sky_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test_noRot.csv')


SDSS = pd.merge(SDSS_orig, SDSS_shuffled, on='galcount', suffixes=('','_shuffled'))
TNG = pd.merge(TNG_orig, TNG_shuffled, on='objid', suffixes=('','_shuffled'))
Illustris = pd.merge(Illustris_orig, Illustris_shuffled, on='objid', suffixes=('','_shuffled'))
blobs = pd.merge(blobs_orig, blobs_shuffled, on='objid', suffixes=('','_shuffled'))
sky = pd.merge(sky_orig, sky_shuffled, on='objid', suffixes=('','_shuffled'))
sky_norot = pd.merge(sky_orig_norot, sky_shuffled_norot, on='objid', suffixes=('','_shuffled'))

SDSS['LLR'] = SDSS['likelihood'] - SDSS['likelihood_shuffled']
TNG['LLR'] = TNG['likelihood'] - TNG['likelihood_shuffled']
Illustris['LLR'] = Illustris['likelihood'] - Illustris['likelihood_shuffled']
#noise['LLR'] = noise['likelihood'] - noise['likelihood_shuffled']
blobs['LLR'] = blobs['likelihood_shuffled']-blobs['likelihood']
sky['LLR'] = sky['likelihood_shuffled']-sky['likelihood']
sky_norot['LLR'] = sky_norot['likelihood_shuffled']-sky_norot['likelihood']



SDSS.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/SDSS_Rot_blobsLike.csv', index=False)
TNG.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG_Rot_blobsLike.csv', index=False)
Illustris.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/Illustris_Rot_blobsLike.csv', index=False)
blobs.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/blobs_Rot_blobsLike.csv', index=False)
sky.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/sky_Rot_blobsLike.csv', index=False)
sky_norot.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/sky_noRot_blobsLike.csv', index=False)


fig,ax = plt.subplots(1,1)
bins = np.arange(-100,1000,10)
SDSS['LLR'].hist(bins=bins,ax=ax, histtype='step', label='SDSS', color='darkorange', lw=5, density=True)
TNG['LLR'].hist(bins=bins,ax=ax, histtype='step', label='TNG', color='teal', lw=5, ls='--', density=True)
Illustris['LLR'].hist(bins=bins,ax=ax, histtype='step', label='Illustris', color='firebrick', lw=5, ls=':', density=True)
blobs['LLR'].hist(bins=bins,ax=ax, color='limegreen',lw=2, label='blobs', normed=True, histtype='step')
sky['LLR'].hist(bins=bins,ax=ax, histtype='step', label='sky', color='deeppink', lw=5, ls='-.', density=True)
sky_norot['LLR'].hist(bins=bins,ax=ax, histtype='step', label='sky, no rotation', color='gold', lw=2, ls='--', density=True)
plt.legend(fontsize=25, frameon=False)
#plt.ylim(0,0.01)
plt.xlabel('LLR')
plt.ylabel('#')
plt.xlim(-100,420)
plt.tight_layout()
plt.savefig('./results/likelihood_plots/LLR.png')
plt.close()

fig,ax = plt.subplots(1,1)
bins = np.arange(3000,4500,10)
SDSS['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='SDSS', color='darkorange', lw=5, density=True)
TNG['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='TNG', color='teal', lw=5, ls='--', density=True)
Illustris['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='Illustris', color='firebrick', lw=5, ls=':', density=True)
blobs['likelihood'].hist( bins=bins, ax=ax, color='limegreen', lw=2,label='blobs', histtype='step', normed=True)
sky['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='sky', color='deeppink', lw=5, ls='-.', density=True)
sky_norot['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='sky, no rotation', color='gold', lw=2, ls='--', density=True)
plt.legend(fontsize=20, frameon=False, loc='upper left')
plt.ylim(0,0.005)
plt.xlabel('likelihood - baseline model')
plt.ylabel('#')
plt.tight_layout()
plt.savefig('./results/likelihood_plots/likelihood_original.png')
plt.close()

fig,ax = plt.subplots(1,1)
bins = np.arange(3000,4500,10)
SDSS['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='SDSS', color='darkorange', lw=5, density=True)
TNG['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='TNG', color='teal', lw=5, ls='--', density=True)
Illustris['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='Illustris', color='firebrick', lw=5, ls=':', density=True)
blobs['likelihood_shuffled'].hist(bins=bins, ax=ax,color='limegreen', lw=2, label='blobs', histtype='step', normed=True)
sky['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='sky', color='deeppink', lw=5, ls='-.', density=True)
sky_norot['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='sky, no rotation', color='gold', lw=2, ls='--', density=True)
plt.legend(fontsize=20, frameon=False, loc='upper left')
plt.ylim(0,0.005)
plt.xlabel('likelihood - blobs model')
plt.ylabel('#')
plt.tight_layout()
plt.savefig('./results/likelihood_plots/likelihood_blobs.png')


