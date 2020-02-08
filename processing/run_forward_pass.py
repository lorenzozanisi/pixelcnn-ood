import forward_pass as fp

path = '/scratch/lzanisi/pixel-cnn/'
model1 =path+'trained/0.02_0.08/asinh_SDSS_blobsLike/1Msteps/pixelcnn_out'
model2 = path+'trained/0.02_0.08/asinh_NewSersicBlobs_SerOnly/1Msteps/pixelcnn_out'

inputfile = path+'Empty_fields/empty_fields.pkl'
df = path+'data/df_empty_fields.dat'
pipeline = fp.LikelihoodPipeline(get_locscale=True)
pipeline.process(inputfile, df, dataset='sky', model1=model1, model2=model2)

InputFiles = [path+'sersic_blobs/sersic_newblobs_SerOnly_skysub_FullReal_Uncalibrated_test.pkl',
         path+'SDSS_cutouts/0.02_0.08_Mstar_gt10_asinh_test_blobsLike.pkl',
             path+'Illustris1/0.045_skysub.pkl',
             path+'TNG_cutouts/0.045_skysub.pkl',
             path+'Empty_fields/empty_fields.pkl']
'''
DataSets = ['blobs','SDSS','Illustris','TNG','sky']
InputDataFrames = [path+'data/df_cleaned_goodBlobs_SerOnly.dat',
                   path+'data/cleaned_df_bis_0.02_0.08_blobsLike.dat',
                   path+'data/df_ill_ordered_Mgt10.csv',
                   path+'data/df_TNG_ordered_Mgt10.csv',
                  path+'data/df_empty_fields.dat']

pipeline = fp.LikelihoodPipeline(get_locscale=True)
for file, dset, df in zip(InputFiles, DataSets,InputDataFrames):
    print(dset)
    pipeline.process(file, df, dset, model1=model1, model2=model2)
'''
    
    
    
    
