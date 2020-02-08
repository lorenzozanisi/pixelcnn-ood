import numpy as np
from maps import data_loader, plotter
import matplotlib.pylab as plt
import os
path = '/scratch/lzanisi/pixel-cnn/'

def make_dirs(model):
    name = model.split('/')[-2]
    run = model.split('/')[-3]
    if not os.path.exists(path+'analysis/results/{}_{}'.format(name,run)):
        os.mkdir(path+'analysis/results/{}_{}'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/SDSS'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/SDSS/likelihood'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/SDSS/LLR'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/SDSS/sigma'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/SDSS/mu'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/SDSS/data'.format(name,run))
        
        os.mkdir(path+'analysis/results/{}_{}/explore/TNG'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/TNG/likelihood'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/TNG/LLR'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/TNG/sigma'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/TNG/mu'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/TNG/data'.format(name,run))
        
        os.mkdir(path+'analysis/results/{}_{}/explore/Illustris'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/Illustris/likelihood'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/Illustris/LLR'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/Illustris/sigma'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/Illustris/mu'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/Illustris/data'.format(name,run))
        
        os.mkdir(path+'analysis/results/{}_{}/explore/sky'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/sky/likelihood'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/sky/LLR'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/sky/sigma'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/sky/mu'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/explore/sky/data'.format(name,run))

        os.mkdir(path+'analysis/results/{}_{}/outliers'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/SDSS'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/SDSS/likelihood'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/SDSS/LLR'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/SDSS/sigma'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/SDSS/mu'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/SDSS/data'.format(name,run))
        
        os.mkdir(path+'analysis/results/{}_{}/outliers/TNG'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/TNG/likelihood'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/TNG/LLR'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/TNG/sigma'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/TNG/mu'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/TNG/data'.format(name,run))
        
        os.mkdir(path+'analysis/results/{}_{}/outliers/Illustris'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/Illustris/likelihood'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/Illustris/LLR'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/Illustris/sigma'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/Illustris/mu'.format(name,run))
        os.mkdir(path+'analysis/results/{}_{}/outliers/Illustris/data'.format(name,run))
    
def get_maps(dataset, model1, model2):
    
    loader = data_loader(dataset=dataset, model=model1,model2=model2)
    LLR, ids = loader.load(map_type='LLR')
    L, _ = loader.load(map_type='likelihoodmap')
    data, _ = loader.load(map_type='data_rot')
    mu, _ = loader.load(map_type='locmap')
    sigma, _ =loader.load(map_type='scalemap')
    
    loader2 = data_loader(dataset=dataset, model=model2,model2=model1)
    LLR2, ids2 = loader2.load(map_type='LLR')
    L2, _ = loader2.load(map_type='likelihoodmap')
    data2, _ = loader2.load(map_type='data_rot')
    mu2, _ = loader2.load(map_type='locmap')
    sigma2, _ =loader2.load(map_type='scalemap')
    
    plot = plotter(dataset=dataset)
    if dataset!='sky':
        plot.get_maps(data=data, mu=mu,sigma=sigma, lmap=L, llrmap=LLR, model=model1,\
                  data2=data2, mu2=mu2,sigma2=sigma2, lmap2=L2, llrmap2=LLR2, model2=model2,  action='outliers')
    plot.get_maps(data=data, mu=mu,sigma=sigma, lmap=L, llrmap=LLR, model=model1,\
              data2= data2, mu2=mu2,sigma2=sigma2, lmap2=L2, llrmap2=LLR2, model2=model2,  action='explore')
    
if __name__=='__main__':
    model1 =path+'trained/0.02_0.08/asinh_SDSS_blobsLike/1Msteps/pixelcnn_out'
    model2 = path+'trained/0.02_0.08/asinh_NewSersicBlobs_SerOnly/1Msteps/pixelcnn_out'    
    make_dirs(model1)
    make_dirs(model2)
    
    get_maps(dataset="blobs", model1=model1, model2=model2)
