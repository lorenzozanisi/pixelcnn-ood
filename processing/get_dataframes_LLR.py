import pandas as pd
import numpy as np


SDSS_orig = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/SDSS_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test.csv')
SDSS_orig_train = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/SDSS_df_likelihood_asinh_SDSS_blobsLike_1Msteps_train.csv')

TNG_orig = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/TNG_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test.csv')
Illustris_orig = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/Illustris_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test.csv')
#noise_orig = pd.read_csv('noisetest_df_likelihood_asinh_Mgt10_3resnets_rightOrder_800ksteps.csv',
#                        delim_whitespace=True)
blobs_orig = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/blobs_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test.csv')
blobs_train = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/blobs_df_likelihood_asinh_SDSS_blobsLike_1Msteps_train.csv')
sky_orig = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/sky_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test.csv')
sky_orig_norot = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/sky_df_likelihood_asinh_SDSS_blobsLike_1Msteps_test_noRot.csv')


SDSS_shuffled = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/SDSS_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test.csv')
SDSS_shuffled_train = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/SDSS_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_train.csv')

TNG_shuffled = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/TNG_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test.csv')
Illustris_shuffled = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/Illustris_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test.csv')
#noise_shuffled = pd.read_csv('noisetest_df_likelihood_asinh_SersicBlobs_SerOnly_1Msteps.csv'#,
#                            delim_whitespace=True)
blobs_shuffled = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/blobs_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test.csv')
blobs_shuffled_train = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/blobs_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_train.csv')

sky_shuffled = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/sky_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test.csv')
sky_shuffled_norot = pd.read_csv('/scratch/lzanisi/pixel-cnn/outputs/test_double_model/sky_df_likelihood_asinh_NewSersicBlobs_SerOnly_1Msteps_test_noRot.csv')


SDSS = pd.merge(SDSS_orig, SDSS_shuffled, on='galcount', suffixes=('','_shuffled'))
SDSS_train = pd.merge(SDSS_orig_train, SDSS_shuffled_train, on='galcount', suffixes=('','_shuffled'))

TNG = pd.merge(TNG_orig, TNG_shuffled, on='objid', suffixes=('','_shuffled'))
Illustris = pd.merge(Illustris_orig, Illustris_shuffled, on='objid', suffixes=('','_shuffled'))
blobs = pd.merge(blobs_orig, blobs_shuffled, on='objid', suffixes=('','_shuffled'))
blobs_train = pd.merge(blobs_orig, blobs_shuffled_train, on='objid', suffixes=('','_shuffled'))

sky = pd.merge(sky_orig, sky_shuffled, on='objid', suffixes=('','_shuffled'))
sky_norot = pd.merge(sky_orig_norot, sky_shuffled_norot, on='objid', suffixes=('','_shuffled'))

SDSS['LLR'] = SDSS['likelihood'] - SDSS['likelihood_shuffled']

SDSS_train['LLR'] = SDSS_train['likelihood'] - SDSS_train['likelihood_shuffled']
TNG['LLR'] = TNG['likelihood'] - TNG['likelihood_shuffled']
Illustris['LLR'] = Illustris['likelihood'] - Illustris['likelihood_shuffled']
#noise['LLR'] = noise['likelihood'] - noise['likelihood_shuffled']
blobs_train['LLR'] = blobs_train['likelihood_shuffled']-blobs_train['likelihood']
blobs['LLR'] = blobs['likelihood_shuffled']-blobs['likelihood']

sky['LLR'] = sky['likelihood_shuffled']-sky['likelihood']
sky_norot['LLR'] = sky_norot['likelihood_shuffled']-sky_norot['likelihood']



SDSS.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/SDSS_Rot_blobsLike.csv', index=False)
SDSS_train.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/SDSS_Rot_blobsLike_train.csv', index=False)
TNG.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/TNG_Rot_blobsLike.csv', index=False)
Illustris.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/Illustris_Rot_blobsLike.csv', index=False)
blobs.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/blobs_Rot_blobsLike.csv', index=False)
blobs_train.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/blobs_Rot_blobsLike_train.csv', index=False)
sky.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/sky_Rot_blobsLike.csv', index=False)
sky_norot.to_csv('/scratch/lzanisi/pixel-cnn/data/DataFrames_LLR/sky_noRot_blobsLike.csv', index=False)
