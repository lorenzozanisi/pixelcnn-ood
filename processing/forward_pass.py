import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom, rotate
import pickle as pkl

class LikelihoodPipeline:
    def __init__(self, get_locscale=False):
        self.get_locscale = get_locscale
        
    def get_Likelihood(self,data, model, ids):
        pixelcnn_model = hub.Module(model)
        x = tf.placeholder(shape=(1,32,32,1), dtype=tf.float32)
        d = pixelcnn_model(x,as_dict=True)
        l = -d['log_prob']
        if self.get_locscale:
            loc = d['loc']
            scale = d['scale']
            likelihoodmap = d['likelihood_map']
            loc_img = []
            scale_img = []        
            lmap_img = []
        sess= tf.Session()
        sess.run(tf.global_variables_initializer())     
        Likelihood = []
        logprob_pixel = []
        for i,img in enumerate(data):
            print(i)
            img = np.expand_dims(img, axis=0)
            if self.get_locscale:
                L,mu,sigma,lmap = sess.run([l,loc,scale, likelihoodmap], feed_dict={x : img})
                loc_img.append(mu)
                scale_img.append(sigma)
                lmap_img.append(lmap)
            else:
                L = sess.run(l, feed_dict={x : img})
            Likelihood.append(L[0])
        if self.get_locscale:
            loc_img = np.asarray(loc_img).reshape((len(loc_img),32,32))
            scale_img = np.asarray(scale_img).reshape((len(scale_img),32,32))
            lmap_img = np.asarray(lmap_img).reshape((len(lmap_img),32,32))
            data = np.asarray(data).reshape((len(data),32,32))
            return {'likelihood':np.array(Likelihood), 'objid':np.array(ids)},\
                    {'locmap':np.asarray(loc_img),'objid':np.array(ids)},\
                    {'scalemap': np.asarray(scale_img), 'objid':np.array(ids)},\
                     {'likelihoodmap': np.asarray(lmap_img),'objid':np.array(ids)},\
                    {'data_rot':data, 'objid':np.array(ids)}
        else:
            return {'likelhood':np.array(Likelihood), 'objid':np.array(ids)}  
        
    def get_rotate(self,x):
        angle = np.random.uniform(0,180)
        x = rotate(x, angle, reshape=False)
        return x

    def get_zoom(self,x):
        x = zoom(x,0.5)
        return x
    
    def process(self,InputDataFile, InputDF, dataset, model1, suffix='test',
                model2=None):
        '''
        InputDataFile:  string, the .pkl object where the data is stored. 
                        Make sure it is processed in the same way as the training set.
        InputDF: Pandas DataFrame where the data to be matched with the output likelihood is stored
        dataset: str, can be 'SDSS','blobs','TNG','Illustris','sky'
        model1:  str, the first model
        suffix_dfName: str, suffix for the name of the output DataFrame - 
                        use in case you need it to specify whether it is train or test.
                        Default is 'test'
        model2: (optional): str, the second model to be compared with in case of the LLR

        '''
        with open(InputDataFile,'rb') as f:
            obj = pkl.load(f)
        data = np.array(obj['data'])
        objid = np.array(obj['objid'])   # need to change this for SDSS, as it now has galcount and not objid
        print(data.shape)
        print('...performing rotation and crop...')
        data = data + 1.e-4
        data = np.array(list(map(self.get_rotate,data)))
        print('rotated')
        data = data[:,32:96,32:96]  
        data = np.array(list(map(self.get_zoom,data)))
        print('zoomed')
        data = np.expand_dims(data, axis=3)
        
        name1 = model1.split('/')[-2]
        run1 = model1.split('/')[-3]
        
        print('read DF')
        df = pd.read_csv(InputDF, sep=' ')
        print('running')
        if not self.get_locscale: 
            dict_L1 = self.get_Likelihood(data, model1,objid)
            print('matching')
            df_L1 = pd.DataFrame(dict_L1)
            df1 = df_L1.merge(df, on='objid')
            print('saving')
            df1.to_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/{}_df_likelihood_{}_{}_{}.csv'.format(dataset,run1, name1, suffix), index=False)
            if model2 is not None:
                name2 = model2.split('/')[-2]
                run2 = model2.split('/')[-3]        
                dict_L2 = self.get_Likelihood(data, model2,objid)
                print('matching')
                df_L2 = pd.DataFrame(dict_L2)
                df2 = df_L2.merge(df, on='objid')
                print('saving')
                df2.to_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/{}_df_likelihood_{}_{}_{}.csv'.format(dataset,run2, name2, suffix),index=False)    
        else:
            dict_L1, dict_loc, dict_scale,dict_lmap, dict_data = self.get_Likelihood(data, model1,objid)
            df_L1 = pd.DataFrame(dict_L1)
            print('matching 1 ')
            df1 = df_L1.merge(df, on='objid')
            print('saving 1')
            df1.to_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/{}_df_likelihood_{}_{}_{}.csv'.format(dataset,run1, name1, suffix), index=False)
            with open('/scratch/lzanisi/pixel-cnn/outputs/{}/loc_{}_{}_{}.pkl'.format(dataset,run1, name1, suffix),'wb') as f:
                pkl.dump(dict_loc, f, protocol=4)
            with open('/scratch/lzanisi/pixel-cnn/outputs/{}/scale_{}_{}_{}.pkl'.format(dataset,run1, name1, suffix),'wb') as f:
                pkl.dump(dict_scale, f, protocol=4)
            with open('/scratch/lzanisi/pixel-cnn/outputs/{}/lmap_{}_{}_{}.pkl'.format(dataset,run1, name1, suffix),'wb') as f:
                pkl.dump(dict_lmap, f, protocol=4)
            with open('/scratch/lzanisi/pixel-cnn/outputs/{}/data_{}_{}_{}.pkl'.format(dataset,run1, name1, suffix),'wb') as f:
                pkl.dump(dict_data, f, protocol=4)

            if model2 is not None:
                name2 = model2.split('/')[-2]
                run2 = model2.split('/')[-3]        
                dict_L2, dict_loc, dict_scale,dict_lmap, dict_data = self.get_Likelihood(data, model2,objid)
                print('matching 2')
                df_L2 = pd.DataFrame(dict_L2)
                df2 = df_L2.merge(df, on='objid')
                print('saving 2')
                df2.to_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/{}_df_likelihood_{}_{}_{}.csv'.format(dataset,run2, name2, suffix),index=False)    
                print('saving images')
                with open('/scratch/lzanisi/pixel-cnn/outputs/{}/loc_{}_{}_{}.pkl'.format(dataset,run2, name2, suffix),'wb') as f:
                    pkl.dump(dict_loc, f, protocol=4)
                with open('/scratch/lzanisi/pixel-cnn/outputs/{}/scale_{}_{}_{}.pkl'.format(dataset,run2, name2, suffix),'wb') as f:
                    pkl.dump(dict_scale, f, protocol=4)
                with open('/scratch/lzanisi/pixel-cnn/outputs/{}/lmap_{}_{}_{}.pkl'.format(dataset,run2, name2, suffix),'wb') as f:
                    pkl.dump(dict_lmap, f, protocol=4)
                with open('/scratch/lzanisi/pixel-cnn/outputs/{}/data_{}_{}_{}.pkl'.format(dataset,run2, name2, suffix),'wb') as f:
                    pkl.dump(dict_data, f, protocol=4)    
        return