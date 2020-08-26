import seaborn as sns
from astropy.io import fits
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.stats import binned_statistic_2d
from scipy import interpolate
from astropy.table import Table
from utils import load_datasets
import matplotlib
plt.rcParams['axes.linewidth']= 3.
plt.rcParams['axes.titlepad'] = 20
plt.rcParams['axes.linewidth']=5
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
plt.rcParams['font.size'] = 40
plt.rcParams['figure.figsize'] = (16,16)
matplotlib.use('Agg')


def make_bootstrap(choice='lowmass', boxes=None, bins=None):

    
    if choice=='lowmass':
            binsM = np.linspace(9.5,10.1,6)
    if choice=='medmass':
            binsM = np.linspace(10.,10.6,6)
    if choice =='highmass':
            binsM = np.linspace(10.5,11.3,8)
    if choice =='all':
            binsM = np.arange(9.5,11.4,0.1)
            
    binsR = np.linspace(-0.5,1.5,10)
    binsN = np.linspace(0,6,6)

    Nboot = 50
    
    LLR_TNG = []
    LLR_TNG50 = []
    LLR_TNG50_2 = []
    LLR_TNG50_3 = []
    LLR_SDSS = []
    LLR_Ill = []
    
    boot_mean_TNG = []
    boot_mean_TNG50 = []
    boot_mean_TNG50_2 = []
    boot_mean_TNG50_3 = []
    boot_mean_SDSS = []
    boot_mean_Ill = []
    for n in range(Nboot):
        print(f'bootstrap step: {n}')
        TNG50_3_ = TNG50_3[['mag','Re','n_bulge','LLR','sSFR']].copy()#query('-50<LLR<300 & sSFR<-11').copy()
        TNG50_2_ = TNG50_2[['mag','Re','n_bulge','LLR','sSFR']].copy()#query('-50<LLR<300 & sSFR<-11').copy()
        TNG50_ = TNG50[['mag','Re','n_bulge','LLR','sSFR']].copy()#query('sSFR<-11').copy()
       # TNG_ = TNG[['Mstar','Re','n_bulge','LLR']].query('-50<LLR<300').copy()
        SDSS_ = SDSS[['mag','Re','n_bulge','LLR','sSFR']].copy()#query('sSFR<-11').copy()
        Ill_ = Illustris[['mag','Re','n_bulge','LLR','sSFR']].copy()#query('sSFR<-11').copy()
        
        arr_TNG50_3 = []#pd.DataFrame()
        arr_TNG50_2 = []#pd.DataFrame()
        arr_TNG50 = []#pd.DataFrame()
        arr_TNG = []#pd.DataFrame()
        arr_SDSS = []#pd.DataFrame()
        arr_Ill = []#pd.DataFrame()

        for i in range(len(binsM)-1):
            for j in range(len(binsR)-1):
                for k in range(len(binsN)-1):
                    app_box = []
                    Sbox = []
                    for box in boxes:
                        B = box.query(f'{binsM[i]}<mag<{binsM[i+1]}  & {binsR[j]}<Re<{binsR[j+1]} & {binsN[k]}<n_bulge<{binsN[k+1]}')
                        Sbox.append(B)
                        app_box.append(len(B))


                    S50_3 = TNG50_3_.query(f'{binsM[i]}<mag<{binsM[i+1]} & {binsR[j]}<Re<{binsR[j+1]} & {binsN[k]}<n_bulge<{binsN[k+1]}')
                    S50_2 = TNG50_2_.query(f'{binsM[i]}<mag<{binsM[i+1]} & {binsR[j]}<Re<{binsR[j+1]} & {binsN[k]}<n_bulge<{binsN[k+1]}')
                    S50 = TNG50_.query(f'{binsM[i]}<mag<{binsM[i+1]} & {binsR[j]}<Re<{binsR[j+1]} & {binsN[k]}<n_bulge<{binsN[k+1]}')
                    S = SDSS_.query(f'{binsM[i]}<mag<{binsM[i+1]}  & {binsR[j]}<Re<{binsR[j+1]} & {binsN[k]}<n_bulge<{binsN[k+1]}')
                    S_Ill = Ill_.query(f'{binsM[i]}<mag<{binsM[i+1]}  & {binsR[j]}<Re<{binsR[j+1]} & {binsN[k]}<n_bulge<{binsN[k+1]}')

                    boxMin = np.min(app_box)
                    S_min = np.min([len(S50_3),len(S50_2),len(S50),len(S), len(S_Ill), boxMin]) 
                    if S_min>0:    

                                  #sample the minimum number of objects
                        S50 = S50.sample(n=S_min)#.query('sSFR>-11')
                        S50_2 = S50_2.sample(n=S_min)#.query('sSFR>-11')
                        S50_3 = S50_3.sample(n=S_min)#.query('sSFR>-11')             
                        S = S.sample(n=S_min)#.query('sSFR>-11')
                        S_Ill = S_Ill.sample(n=S_min)#.query('sSFR>-11')
                        for B in Sbox :     #sample al the boxes as well, TNG100 is the collection of the 8 boxes
                            B = B.sample(n=S_min)#.query('sSFR>-11')
                            arr_TNG.extend(B['LLR'])

                        arr_TNG50.extend(S50['LLR'])
                        arr_TNG50_2.extend(S50_2['LLR'])
                        arr_TNG50_3.extend(S50_3['LLR'])
                        arr_SDSS.extend(S['LLR'])
                        arr_Ill.extend(S_Ill['LLR'])

                        
        LLR_TNG.append(np.histogram(arr_TNG, bins=bins, density=True)[0])
        LLR_TNG50.append(np.histogram(arr_TNG50, bins=bins, density=True)[0])
        LLR_TNG50_2.append(np.histogram(arr_TNG50_2, bins=bins, density=True)[0])
        LLR_TNG50_3.append(np.histogram(arr_TNG50_3, bins=bins, density=True)[0])
        LLR_SDSS.append(np.histogram(arr_SDSS, bins=bins, density=True)[0])
        LLR_Ill.append(np.histogram(arr_Ill, bins=bins, density=True)[0])
        
        boot_mean_TNG.append(np.mean(arr_TNG))
        boot_mean_TNG50.append(np.mean(arr_TNG50))
        boot_mean_TNG50_2.append(np.mean(arr_TNG50_2))
        boot_mean_TNG50_3.append(np.mean(arr_TNG50_3))
        boot_mean_SDSS.append(np.mean(arr_SDSS))
        boot_mean_Ill.append(np.mean(arr_Ill))

    return LLR_SDSS, LLR_TNG50, LLR_TNG50_2, LLR_TNG50_3, LLR_TNG, LLR_Ill, boot_mean_SDSS,boot_mean_TNG50,boot_mean_TNG50_2, boot_mean_TNG50_3, boot_mean_TNG,boot_mean_Ill



if __name__=='__main__':
    SDSS, TNG50, TNG, Illustris = load_datasets(orig=True)
    TNG50_dustless = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG50_dustless_Rot_blobsLike_orig_0.03_0.055_new.csv')
    TNG50_2 = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG50_2_Rot_blobsLike_orig_0.03_0.055_new.csv')
    TNG50_3 = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG50_3_Rot_blobsLike_orig_0.03_0.055_new.csv')

    box1 = TNG.query('X<37000 & Y<37000 & Z<37000')[['Mstar','Re','n_bulge','LLR','sSFR']].query('-50<LLR<300')# & sSFR<-11')
    box2 = TNG.query('37000<X & Y<37000 & Z<37000')[['Mstar','Re','n_bulge','LLR','sSFR']].query('-50<LLR<300 ')#& sSFR<-11')
    box3 = TNG.query('X<37000 & 37000<Y & Z<37000')[['Mstar','Re','n_bulge','LLR','sSFR']].query('-50<LLR<300 ')#& sSFR<-11')
    box4 = TNG.query('X<37000 & Y<37000 & 37000<Z')[['Mstar','Re','n_bulge','LLR','sSFR']].query('-50<LLR<300 ')#& sSFR<-11')
    box5 = TNG.query('37000<X & 37000<Y & Z<37000')[['Mstar','Re','n_bulge','LLR','sSFR']].query('-50<LLR<300 ')#& sSFR<-11')
    box6 = TNG.query('X<37000 & 37000<Y & 37000<Z')[['Mstar','Re','n_bulge','LLR','sSFR']].query('-50<LLR<300 ')#& sSFR<-11')
    box7 = TNG.query('37000<X & Y<37000 & 37000<Z')[['Mstar','Re','n_bulge','LLR','sSFR']].query('-50<LLR<300')# & sSFR<-11')
    box8 = TNG.query('37000<X & 37000<Y & 37000<Z')[['Mstar','Re','n_bulge','LLR','sSFR']].query('-50<LLR<300')# & sSFR<-11')

    boxes = [box1,box2,box3,box4,box5,box6,box7,box8]

    bins = np.arange(-50,400,10) #LLR bins
    
    B = np.arange(10,250,1)
    fig,(ax1,ax2,ax3,ax4 ) = plt.subplots(1,4, figsize=(64,16))
    fig_1,(ax1_1,ax2_1,ax3_1 ) = plt.subplots(1,3, figsize=(48,16))
    fig_2, Ax = plt.subplots(1,1, figsize=(16,16))
    
    LLR_SDSS, LLR_TNG50, LLR_TNG50_2, LLR_TNG50_3, LLR_TNG, LLR_Ill, boot_mean_SDSS,boot_mean_TNG50,boot_mean_TNG50_2, boot_mean_TNG50_3, boot_mean_TNG,boot_mean_Ill = make_bootstrap(choice='lowmass', boxes=boxes, bins=bins)
    ys = 0.012
    ax1_1.text(125,ys,r'$\Delta \langle LLR \rangle$=')
    for hists, (means,lab, col, facecol,ls) in zip([LLR_SDSS, LLR_TNG50, LLR_TNG50_2, LLR_TNG50_3, LLR_TNG, LLR_Ill],\
                                            zip([boot_mean_SDSS,boot_mean_TNG50,boot_mean_TNG50_2, boot_mean_TNG50_3, boot_mean_TNG, boot_mean_Ill],\
                                            ['SDSS','TNG50','TNG50-2','TNG50-3','TNG100','Illustris'],\
                                         ['darkorange','magenta','darkgray','darkolivegreen','teal','firebrick'],['moccasin','pink','lightgray','green','cyan','salmon'],\
                                            ['-','--',':','-.','-',':'])):

            H = np.histogram(means, bins=B, density=True)[0]
            ax1.plot(B[1:]-0.5,H, label=lab, color=col, lw=6, ls=ls)

            
            if lab=='SDSS':    
                low, med, up = np.percentile(np.array(hists).T, [16,50,84], axis=1)
                ax1_1.plot(bins[1:],med, label=lab, color=col, lw=4, ls=ls)
                med_SDSS = means.copy()
            else:
                low, med, up = np.percentile(np.array(hists).T, [16,50,84], axis=1)
                ax1_1.plot(bins[1:],med, label=lab, color=col, lw=4, ls=ls)
                std = np.sqrt(np.std(means)**2 + np.std(med_SDSS)**2 )
                delta = np.mean(means)-np.mean(med_SDSS)
                ax1_1.text(230,ys,r'${} \pm {}$'.format(np.round(delta,2),np.round(std,2)),color=col)
                ys = ys-0.0025            
            
            
    LLR_SDSS, LLR_TNG50, LLR_TNG50_2, LLR_TNG50_3, LLR_TNG, LLR_Ill, boot_mean_SDSS,boot_mean_TNG50,boot_mean_TNG50_2, boot_mean_TNG50_3, boot_mean_TNG,boot_mean_Ill = make_bootstrap(choice='medmass', boxes=boxes, bins=bins)
    ys = 0.012
    ax2_1.text(125,ys,r'$\Delta \langle LLR \rangle$=')
    for hists, (means,lab, col, facecol,ls) in zip([LLR_SDSS, LLR_TNG50, LLR_TNG50_2, LLR_TNG50_3, LLR_TNG, LLR_Ill],\
                                            zip([boot_mean_SDSS,boot_mean_TNG50,boot_mean_TNG50_2, boot_mean_TNG50_3, boot_mean_TNG, boot_mean_Ill],\
                                            ['SDSS','TNG50','TNG50-2','TNG50-3','TNG100','Illustris'],\
                                         ['darkorange','magenta','darkgray','darkolivegreen','teal','firebrick'],['moccasin','pink','lightgray','green','cyan','salmon'],\
                                            ['-','--',':','-.','-',':'])):

            H = np.histogram(means, bins=B, density=True)[0]
            ax2.plot(B[1:]-0.5,H, label=lab, color=col, lw=6, ls=ls)

            
            if lab=='SDSS':    
                low, med, up = np.percentile(np.array(hists).T, [16,50,84], axis=1)
                ax2_1.plot(bins[1:],med, label=lab, color=col, lw=4, ls=ls)
                med_SDSS = means.copy()
            else:
                low, med, up = np.percentile(np.array(hists).T, [16,50,84], axis=1)
                ax2_1.plot(bins[1:],med, label=lab, color=col, lw=4, ls=ls)
                std = np.sqrt(np.std(means)**2 + np.std(med_SDSS)**2 )
                delta = np.mean(means)-np.mean(med_SDSS)
                ax2_1.text(230,ys,r'${} \pm {}$'.format(np.round(delta,2),np.round(std,2)),color=col)
                ys = ys-0.0025            
                
                
    LLR_SDSS, LLR_TNG50, LLR_TNG50_2, LLR_TNG50_3, LLR_TNG, LLR_Ill, boot_mean_SDSS,boot_mean_TNG50,boot_mean_TNG50_2, boot_mean_TNG50_3, boot_mean_TNG,boot_mean_Ill = make_bootstrap(choice='highmass', boxes=boxes, bins=bins)
    ys = 0.012
    ax3_1.text(125,ys,r'$\Delta \langle LLR \rangle$=')
    for hists, (means,lab, col, facecol,ls) in zip([LLR_SDSS, LLR_TNG50, LLR_TNG50_2, LLR_TNG50_3, LLR_TNG, LLR_Ill],\
                                            zip([boot_mean_SDSS,boot_mean_TNG50,boot_mean_TNG50_2, boot_mean_TNG50_3, boot_mean_TNG, boot_mean_Ill],\
                                            ['SDSS','TNG50','TNG50-2','TNG50-3','TNG100','Illustris'],\
                                         ['darkorange','magenta','darkgray','darkolivegreen','teal','firebrick'],['moccasin','pink','lightgray','green','cyan','salmon'],\
                                            ['-','--',':','-.','-',':'])):


            H = np.histogram(means, bins=B, density=True)[0]
            ax3.plot(B[1:]-0.5,H, label=lab, color=col, lw=6, ls=ls)

            if lab=='SDSS':    
                low, med, up = np.percentile(np.array(hists).T, [16,50,84], axis=1)
                ax3_1.plot(bins[1:],med, label=lab, color=col, lw=4, ls=ls)
                med_SDSS = means.copy()
            else:
                low, med, up = np.percentile(np.array(hists).T, [16,50,84], axis=1)
                ax3_1.plot(bins[1:],med, label=lab, color=col, lw=4, ls=ls)
                std = np.sqrt(np.std(means)**2 + np.std(med_SDSS)**2 )
                delta = np.mean(means)-np.mean(med_SDSS)
                ax3_1.text(230,ys,r'${} \pm {}$'.format(np.round(delta,2),np.round(std,2)),color=col)
                ys = ys-0.0025            
                
                
    LLR_SDSS, LLR_TNG50, LLR_TNG50_2, LLR_TNG50_3, LLR_TNG, LLR_Ill, boot_mean_SDSS,boot_mean_TNG50,boot_mean_TNG50_2, boot_mean_TNG50_3, boot_mean_TNG,boot_mean_Ill = make_bootstrap(choice='all', boxes=boxes, bins=bins)
    ys = 0.012
    Ax.text(125,ys,r'$\Delta \langle LLR \rangle$=')
    for hists, (means,lab, col, facecol,ls) in zip([LLR_SDSS, LLR_TNG50, LLR_TNG50_2, LLR_TNG50_3, LLR_TNG, LLR_Ill],\
                                            zip([boot_mean_SDSS,boot_mean_TNG50,boot_mean_TNG50_2, boot_mean_TNG50_3, boot_mean_TNG, boot_mean_Ill],\
                                            ['SDSS','TNG50','TNG50-2','TNG50-3','TNG100','Illustris'],\
                                         ['darkorange','magenta','darkgray','darkolivegreen','teal','firebrick'],['moccasin','pink','lightgray','green','cyan','salmon'],\
                                            ['-','--',':','-.','-',':'])):


            H = np.histogram(means, bins=B, density=True)[0]
            ax4.plot(B[1:]-0.5,H, label=lab, color=col, lw=6, ls=ls)
            if lab=='SDSS':    
                low, med, up = np.percentile(np.array(hists).T, [16,50,84], axis=1)
                Ax.plot(bins[1:],med, label=lab, color=col, lw=4, ls=ls)
                med_SDSS = means.copy()
            else:
                low, med, up = np.percentile(np.array(hists).T, [16,50,84], axis=1)
                Ax.plot(bins[1:],med, label=lab, color=col, lw=4, ls=ls)
                std = np.sqrt(np.std(means)**2 + np.std(med_SDSS)**2 )
                delta = np.mean(means)-np.mean(med_SDSS)
                Ax.text(230,ys,r'${} \pm {}$'.format(np.round(delta,2),np.round(std,2)),color=col)
                ys = ys-0.0025            
                
                
    ax1.set_title(r'$9.5<\log{M_{\rm star}/M_\odot}<10$')
    ax2.set_title(r'$10<\log{M_{\rm star}/M_\odot}<10.5$')
    ax3.set_title(r'$\log{M_{\rm star}/M_\odot}>10.5$')
    ax4.set_title('all galaxies')

    #ax1.legend(frameon=False, fontsize=45)
    ax1.legend(frameon=False, fontsize=25)
    #ax.set_ylim(0,0.025)
    ax1.set_xlabel('bootstrap <LLR>')
    ax2.set_xlabel('bootstrap <LLR>')
    ax3.set_xlabel('bootstrap <LLR>')
    ax4.set_xlabel('bootstrap <LLR>')
    ax1.set_ylabel('#')
    
    ax1_1.set_title(r'$9.5<\log{M_{\rm star}/M_\odot}<10$')
    ax2_1.set_title(r'$10<\log{M_{\rm star}/M_\odot}<10.5$')
    ax3_1.set_title(r'$\log{M_{\rm star}/M_\odot}>10.5$')

    #ax1.legend(frameon=False, fontsize=45)
    ax1_1.legend(frameon=False, fontsize=25)
    #ax.set_ylim(0,0.025)
    ax1_1.set_xlabel('LLR')
    ax2_1.set_xlabel('LLR')
    ax3_1.set_xlabel('LLR')
    ax1_1.set_ylabel('#')
    ax1_1.set_ylim(0,0.025)
    #ax3_1.text(200, 0.02,'star forming', fontsize=45)
    
   # fig_2.suptitle('star forming')
    Ax.set_xlabel('LLR')
    Ax.set_ylabel('#')
    Ax.legend(frameon=False, fontsize=25)
    fig.savefig('./bootstraps/matched_Mag-n-R_bootstrapDistribs.pdf', bbox_inches='tight')
    fig_1.savefig('./bootstraps/matched_Mag-n-R_bootstrapped_LLR_masses.pdf', bbox_inches='tight')
    fig_2.savefig('./bootstraps/matched_Mag-n-R_bootstrapped_LLR_all.pdf', bbox_inches='tight')