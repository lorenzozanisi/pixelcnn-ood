import pickle as pkl
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
plt.rcParams['font.size']=45
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
plt.rcParams['axes.titlepad'] = 20 
plt.rcParams['figure.figsize']=(16,12)

class data_loader:
    
    def __init__(self,  dataset, model, model2=None):
        '''
        dataset : string, either "TNG", "Illustris", "SDSS", "sky" or "blobs"
        '''
        self.dataset = dataset
        self.model = model
        if model2 is not None:
            self.model2 = model2
            
    def load(self,map_type):
        '''map_type: string, the directory where the data will be saved. It can be "mu", "sigma","LLR", "likelihood","data"
            returns: array of images (sample_size, width,height), array of images ids (sample_size)
        '''
        dictionary_conversion = {'locmap':'loc', 'data_rot':'data', 'likelihoodmap':'lmap', 'scalemap':'scale', 'LLR':'LLR'}  #this is needed because the naming is different in the pkl files - ToDo: fix this in the data
        self.name_map_type = dictionary_conversion[map_type]     
        if self.name_map_type=='LLR':
            if self.model2 is None:
                raise ValueError("model2 MUST be defined if map_type='LLR'") 

        name = self.model.split('/')[-2]
        run = self.model.split('/')[-3]
        
        if map_type!='LLR':
            with open('/scratch/lzanisi/pixel-cnn/outputs/{}/{}_{}_{}_test.pkl'.format(self.dataset,self.name_map_type,run,name),'rb') as f:
                obj = pkl.load(f)
            data = np.asarray(obj[map_type])
            ids = np.asarray(obj['objid'])
            return data, ids
        else:
            with open('/scratch/lzanisi/pixel-cnn/outputs/{}/lmap_{}_{}_test.pkl'.format(self.dataset,run,name),'rb') as f:
                obj = pkl.load(f)
            data = np.asarray(obj['likelihoodmap'])
            ids = np.asarray(obj['objid'])
            name2 = self.model2.split('/')[-2]
            run2 = self.model2.split('/')[-3]
            with open('/scratch/lzanisi/pixel-cnn/outputs/{}/lmap_{}_{}_test.pkl'.format(self.dataset,run2,name2),'rb') as f:
                obj2 = pkl.load(f)
            data2 = np.asarray(obj2['likelihoodmap'])
            LLR = np.subtract(data,data2)   
            return  LLR, ids
            
            
class plotter: 
    
    def __init__(self,dataset):
        '''
        dataset: string, either "TNG", "Illustris", "SDSS", "sky" or "blobs"
        '''        
        self.df_SDSS = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/SDSS_Rot_blobsLike.csv')
        if dataset!="SDSS":
            self.df_dataset = pd.read_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/{}_Rot_blobsLike.csv'.format(dataset))
            self.dataset = dataset
        else:
            self.dataset = 'SDSS'
        return
    
    def get_outliers_ids(self,choice, low):
        threshold = np.percentile(self.df_SDSS[choice].values, low)
        if self.dataset!= "SDSS":   
            df = self.df_dataset[self.df_dataset[choice].values<threshold]
        else:
            df = self.df_SDSS[self.df_SDSS[choice].values<threshold]
        image_indices = df.index.values  # outliers in the images
        objids = df.objid.values
        return image_indices, objids
    
    def get_choice_ids(self, choice, low, up):
        if self.dataset!= "SDSS": 
            mask = (self.df_dataset[choice].values>low) & (self.df_dataset[choice].values<up)
            df_dataset = self.df_dataset[mask]
            image_indices = df_dataset.index.values 
            objids = df_dataset.objid.values
        else:
            mask = (self.df_SDSS[choice].values>low) & (self.df_SDSS[choice].values<up)
            df_SDSS = self.df_SDSS[mask]
            image_indices = df_SDSS.index.values 
            objids = df_SDSS.objid.values
        return image_indices, objids
    
    def plot(self, data,ids, model, map_type, action="outliers", choice='likelihood', title=None):
        '''
        map_type: string, the directory where the data will be saved. It can be "mu", "sigma","LLR", "likelihood","data".
        action: string, can be "outliers" or "explore". Default is "explore"
        title: string, it's the LLR or likelihood value in the bin when "explore" is chosen. It has no effect if action="outliers"
        '''
        
        name = model.split('/')[-2]
        run = model.split('/')[-3]       
        dic_maps = {'mu': r'$\mu$','sigma':r'$\sigma$','LLR':'LLR','likelihood':'likelihood','data':' '} #object for labelling colorbars
        if action == 'outliers':
            N = int(len(ids)/25)  # 25 plots per figure, N is how many figures there will be
            lim = N*25
            ids = ids[:lim]
            data = data[:lim,:,:]
            for n in range(N):
                fig, ax = plt.subplots(5,5,figsize=(48,48), sharex=True, sharey=True) 
                data_ = data[n*25:(n+1)*25,:,:]
                ids_ = ids[n*25:(n+1)*25]
                for i,this_ax in enumerate(ax.ravel()):
                    try:
                        if map_type=='LLR' :
                            im = this_ax.imshow(data_[i,:,:], cmap='viridis', vmin=-1, vmax=5)
                        elif map_type=='likelihood':
                            im= this_ax.imshow(data_[i,:,:], cmap='viridis', vmin=-1, vmax=5)
                        else:
                            im = this_ax.imshow(data_[i,:,:], cmap='viridis')
                        cbar = plt.colorbar(im, ax=this_ax)
                        cbar.set_label(dic_maps[map_type])
                        this_ax.set_title(ids_[i], fontsize=45)
                        this_ax.tick_params(axis='both',which='both',bottom=False, left=False, labelleft=False, labelbottom=False)
                    except:
                        print('There are less than 25 outliers')
                        pass
                plt.tight_layout()
                plt.savefig('/scratch/lzanisi/pixel-cnn/analysis/results/{}_ranges/{}_{}/outliers/{}/{}/{}.png'.format(choice,name,run,self.dataset,map_type, n))
                plt.close()
            return            
        else:

            fig, ax = plt.subplots(5,5,figsize=(48,48), sharex=True, sharey=True)  #plots only ONE figure, unlike above
            for i,this_ax in enumerate(ax.ravel()):
                if map_type=='LLR' :
                    im = this_ax.imshow(data[i,:,:], cmap='viridis', vmin=-1, vmax=5)
                elif map_type=='likelihood':
                    im= this_ax.imshow(data[i,:,:], cmap='viridis', vmin=-1, vmax=5)
                else:
                    im = this_ax.imshow(data[i,:,:], cmap='viridis')
                cbar = plt.colorbar(im, ax=this_ax)
                cbar.set_label(dic_maps[map_type])
                this_ax.set_title(ids[i], fontsize=45)
                this_ax.tick_params(axis='both',which='both',bottom=False, left=False, labelleft=False, labelbottom=False)
            #fig.suptitle(title)
            plt.tight_layout()
            plt.savefig('/scratch/lzanisi/pixel-cnn/analysis/results/{}_ranges/{}_{}/explore/{}/{}/{}.png'.format(choice,name,run,self.dataset,map_type, title))            
            plt.close()

    def get_maps(self, data, mu,sigma, lmap, llrmap, model, data2=None, mu2=None, sigma2=None, lmap2=None, llrmap2=None, model2=None, choice='likelihood', action='outliers', percentile_threshold=0.15):
        
        '''
        choice : string, either 'likelihood' or 'LLR', sets according to which framework define outliers.     
        data,mu,sigma, lmap, llrmap: they must be loaded with the loader class. It must be the *full* array of data.
        model: string, the trained model used
        action: string, can be either "outliers" or "explore". 
        percentile_threshold (float):  the percentile threshold below which objects are outliers. Default: 3 sigma
        '''
        if model2 is not None:
            if any(i is None for i in [data2,mu2,sigma2,lmap2,llrmap2]):
                raise ValueError('data2,mu2,sigma2,lmap2,llrmap2 must be specified if model2 is not None')
            
        def pipe(ids,objids,data, mu,sigma, lmap, llrmap, model, action, choice='likelihood', title=None, ran = None):
            data = data[ids,:,:] 
            mu = mu[ids,:,:] 
            sigma = sigma[ids,:,:]
            lmap = lmap[ids,:,:]
            LLR = llrmap[ids,:,:]
            if action == 'explore':
                if ran is None:
                    ran = np.random.choice(range(len(ids)), size=25)
                data = data[ran,:,:]
                mu = mu[ran,:,:]
                sigma = sigma[ran,:,:]
                lmap = lmap[ran,:,:]
                LLR = LLR[ran,:,:]
                objids = objids[ran]                
            self.plot(data=data,ids=objids,  model=model, map_type='data', action=action, choice=choice,title=title)
            self.plot(data=mu,ids=objids,  model=model, map_type='mu', action=action, choice=choice,title=title)
            self.plot(data=sigma,ids=objids,  model=model, map_type='sigma', action=action, choice=choice,title=title)
            self.plot(data=LLR,ids=objids, model=model,  map_type='LLR', action=action, choice=choice,title=title)
            self.plot(data=lmap,ids=objids, model=model,  map_type='likelihood', action=action, choice=choice,title=title)
            if action=='explore':
                return ran
            return 0
            
        if action == 'outliers':
            ids, objids = self.get_outliers_ids(choice, percentile_threshold)
            pipe(ids,objids,data, mu,sigma, lmap, llrmap, model, action='outliers',choice=choice)
            if model2 is not None:
                pipe(ids,objids,data2, mu2,sigma2, lmap2, llrmap2, model2, action='outliers',choice=choice)
            print('outliers done')
        else:
            if self.dataset!="SDSS":
                perclow, percup = np.percentile(self.df_dataset[choice], [0.15,99.85]) # 3 sigma
            else:
                perclow, percup = np.percentile(self.df_SDSS[choice], [0.15,99.85])

            nbins = 10
            bins = np.linspace(perclow,percup, nbins)
            for i in range(nbins-1):
                ids, objids = self.get_choice_ids(choice=choice, low=bins[i], up=bins[i+1])
                mean = (bins[i]+bins[i+1])/2.
                ran = pipe(ids,objids,data, mu,sigma, lmap, llrmap, model, action='explore', choice=choice,title=mean) #saves random ids
                if model2 is not None:
                    pipe(ids,objids,data2, mu2,sigma2, lmap2, llrmap2, model2, action='explore', ran=ran, choice=choice,title=mean)
            print('exploration done')
                
        return
            






    
