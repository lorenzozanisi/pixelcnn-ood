import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from scipy.integrate import cumtrapz
import seaborn as sns
plt.style.use('seaborn')
mpl.use('agg')
import pandas as pd
mpl.rcParams['font.size']=45
#mpl.rcParams['xtick.minor.visible']=True

#pl.rcParams['axes.linewidth']= 3.
#mpl.rcParams['axes.titlepad'] = 20
#plt.rcParams['xtick.major.size'] =15
#plt.rcParams['ytick.major.size'] =15
#plt.rcParams['xtick.minor.size'] =10
#plt.rcParams['ytick.minor.size'] =10
#plt.rcParams['xtick.major.width'] =5
#plt.rcParams['ytick.major.width'] =5
#plt.rcParams['xtick.minor.width'] =5
#plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 20 
plt.rcParams['figure.figsize']=(16,12)

SDSS = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/SDSS_Rot_blobsLike_0.03_0.055.csv')
#SDSS_newsky = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/SDSS_newsky_Rot_blobsLike__0.03_0.055.csv')

#SDSS_train = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/SDSS_Rot_blobsLike_train__0.03_0.055.csv')
Illustris = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/Illustris_Rot_blobsLike_orig_0.03_0.055.csv')

TNG = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG_Rot_blobsLike_orig_0.03_0.055.csv')
TNG50 = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG50_Rot_blobsLike_orig_0.03_0.055.csv')

print(TNG50.shape)
blobs = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/blobs_Rot_blobsLike_0.03_0.055.csv')
#blobs_train = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/blobs_Rot_blobsLike_train.csv')



#SDSS = SDSS.query('z>0.04 & z<0.05')
print('bootstrap')
Nboot = 1000
n = len(TNG50)
mean = []
up = []
low = []
LLR_SDSS = []
bins = np.arange(-100,1000,10)
for i in range(Nboot):
    sample = SDSS.sample(n)['LLR']
    LLR_SDSS.append(np.histogram(sample, bins=bins, density=True)[0])
    
print('bbotstrap done')
'''
fig,ax = plt.subplots(1,1)
bins = np.arange(-100,1000,10)

sns.distplot(SDSS['LLR'].values, bins=bins, ax=ax, kde=False)


#ax.grid(which='both',axis='both',linewidth=3, color='darkgray')
SDSS['LLR'].hist(SDSS['LLR'].values, bins=bins,ax=ax, histtype='step', label='SDSS', color='darkorange', lw=7, density=True)
TNG['LLR'].hist(bins=bins,ax=ax, histtype='step', label='TNG', color='teal', lw=7, ls='--', density=True)
Illustris['LLR'].hist(bins=bins,ax=ax, histtype='step', label='Illustris', color='firebrick', lw=7, ls=':', density=True)
blobs['LLR'].hist(bins=bins,ax=ax, color='limegreen',lw=7, label='sersic', normed=True, histtype='step', ls=':')
#sky['LLR'].hist(bins=bins,ax=ax, histtype='step', label='sky', color='deeppink', lw=7, ls='-.', density=True)
#sky_norot['LLR'].hist(bins=bins,ax=ax, histtype='step', label='sky, no rotation', color='gold', lw=2, ls='--', density=True)
#plt.legend(fontsize=30, frameon=False)
#plt.ylim(0,0.01)

plt.xlabel('LLR',fontsize=55)
plt.ylabel('density', fontsize=55)
plt.xlim(-100,420)
plt.tight_layout()
plt.savefig('./results/likelihood_plots/LLR.pdf')
plt.close()



'''

fig,ax = plt.subplots(1,1)
bins = np.arange(-100,1000,10)
#ax.grid(which='both',axis='both',linewidth=3, color='darkgray')


SDSS['LLR'].hist(bins=bins,ax=ax, histtype='step', label='SDSS (real)', color='darkorange', lw=7, density=True)

#low, med, up = np.percentile(np.array(LLR_SDSS).T, [0.15,50,99.85], axis=1)
#ax.plot(bins[1:],med, label='SDSS', color='darkorange', lw=7)
#ax.fill_between(bins[1:], up,low, color='moccasin', alpha=0.8)

TNG['LLR'].hist(bins=bins,ax=ax, histtype='step', label='TNG (simulated)', color='teal', lw=7, ls='--', density=True)
#TNG50['LLR'].hist(bins=bins,ax=ax, histtype='step', label='TNG50', color='magenta', lw=7, ls='--', density=True)

Illustris['LLR'].hist(bins=bins,ax=ax, histtype='step', label='Illustris (simulated)', color='firebrick', lw=7, ls=':', density=True)
#blobs['LLR_real'] = blobs['LLR'].apply(lambda)
#blobs['LLR'].apply(lambda x: -x).hist(bins=bins,ax=ax, color='limegreen',lw=7, label='Archetypes', normed=True, histtype='step', ls='-.')
#SDSS_train['LLR'].hist(bins=bins,ax=ax, histtype='step', label='SDSS - train', color='black', lw=3, density=True)
#sky['LLR'].hist(bins=bins,ax=ax, histtype='step', label='sky', color='deeppink', lw=7, ls='-.', density=True)
#sky_norot['LLR'].hist(bins=bins,ax=ax, histtype='step', label='sky, no rotation', color='gold', lw=2, ls='--', density=True)
#plt.legend(fontsize=30, frameon=False)
plt.legend(fontsize=45, frameon=False, loc='upper right')

plt.ylim(0,0.017)
plt.tick_params(labelsize=45)
plt.xlabel('LLR',fontsize=55)
plt.ylabel('density', fontsize=65)
plt.xlim(-100,420)
plt.tight_layout()
plt.savefig('./results/likelihood_plots/tests/LLR_withTNG50_0.03_0.055_ICLRtalk_Illustris.png')


plt.close()

fig,ax = plt.subplots(1,1)
#ax.grid(which='both',axis='both',linewidth=3, color='darkgray')
bins = np.arange(3000,4500,50)
#SDSS_train['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='SDSS -train', color='black', lw=7, density=True)

SDSS['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='SDSS (real)', color='darkorange', lw=7, density=True)

TNG['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='TNG (simulated)', color='teal', lw=7, ls='--', density=True)

#TNG50['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='TNG50', color='magenta', lw=7, ls='--', density=True)
#Illustris['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='Illustris', color='firebrick', lw=7, ls=':', density=True)
#blobs['likelihood'].hist( bins=bins, ax=ax, color='limegreen', lw=7,label='sersic', histtype='step', normed=True, ls='-.')
#SDSS_train['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='SDSS - train', color='black', lw=3, density=True)

#sky['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='sky', color='deeppink', lw=7, ls='-.', density=True)
#sky_norot['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='sky, no rotation', color='gold', lw=2, ls='--', density=True)
plt.legend(fontsize=45, frameon=False, loc='upper left')

plt.ylim(0,0.005)
plt.tick_params(labelsize=45)
plt.xlabel(r'$log \ likelihood$',fontsize=65)
plt.ylabel('density', fontsize=55)
plt.tight_layout()
plt.savefig('./results/likelihood_plots/tests/likelihood_original_withTNG50_newflux_0.03_0.055_ICLRtalk.png')
plt.close()


# fig,ax = plt.subplots(1,1)
# bins = np.arange(3000,4500,50)

# SDSS['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='SDSS -test', color='darkorange', lw=7, density=True)
# SDSS_train['likelihood'].hist(bins=bins,ax=ax, histtype='step', label='SDSS - train', color='black', lw=3, density=True)
# plt.legend(fontsize=45, frameon=False, loc='upper left')
# plt.ylim(0,0.005)
# plt.tick_params(labelsize=45)
# plt.xlabel(r'$log \ p_{\theta_{SDSS}}$',fontsize=65)
# plt.ylabel('density', fontsize=55)
# plt.tight_layout()
# plt.savefig('./results/likelihood_plots/likelihood_traintest.pdf')
# plt.close()

fig,ax = plt.subplots(1,1)
#ax.grid(which='both',axis='both',linewidth=3, color='darkgray')
bins = np.arange(3000,4500,50)
#blobs_train['likelihood_shuffled'].hist( bins=bins, ax=ax, color='black', lw=7,label='sersic -train', histtype='step', normed=True, ls='-')
SDSS['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='SDSS', color='darkorange', lw=7, density=True)
TNG['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='TNG', color='teal', lw=7, ls='--', density=True)
TNG50['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='TNG50', color='magenta', lw=7, ls='--', density=True)
Illustris['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='Illustris', color='firebrick', lw=7, ls=':', density=True)
blobs['likelihood_shuffled'].hist(bins=bins, ax=ax,color='limegreen', lw=7, label='sersic', histtype='step', normed=True, ls='-.')
#SDSS_train['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='SDSS - train', color='black', lw=3, density=True)
#sky['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='sky', color='deeppink', lw=7, ls='-.', density=True)
#sky_norot['likelihood_shuffled'].hist(bins=bins,ax=ax, histtype='step', label='sky, no rotation', color='gold', lw=2, ls='--', density=True)
#plt.legend(fontsize=20, frameon=False, loc='upper left')
plt.legend(fontsize=45, frameon=False, loc='upper left')

plt.tick_params(labelsize=45)
plt.ylim(0,0.005)
plt.xlabel(r'$log \ p_{\theta_{sersic}}$',fontsize=65)
plt.ylabel('density', fontsize=55)
plt.tight_layout()
plt.savefig('./results/likelihood_plots/tests/likelihood_blobs_withTNG50_newflux_0.03_0.055_magmatch.png')


