import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub
#from tensorflow.python import debug as tf_debug
from pixel_cnn_pp.model import model_spec
#from astropy.visualization import ZScaleInterval
from multiprocessing import Pool
from astropy.io import fits
from scipy.ndimage import rotate, zoom
#from get_data import input_fn
import os
import argparse
import json
#get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
sys.path.insert(0, 'pixel-cnn/')

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--filename', type=str, default='./SDSS_cutouts/filename_0.02_0.08_Mstar_gt10_asinh_blobsLike.txt', help='Location for the dataset')
parser.add_argument('-bat', '--batch_size', type=int, default=64, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
parser.add_argument('-buf','--buffer_size', type=int, default=5000, help='buffer size for shuffling')
parser.add_argument('-cdir','--cache_dir', type=str, default='./cache_asinh/', help='Cache dir')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

image_size = 32
batch_size = args.batch_size
filename = args.filename
buffer_size = args.buffer_size
n_channels = 1
#os.system('rm ./trained/0.02_0.08/*')
os.system('rm -r /tmp/tfdbg*')
os.system('rm -r '+args.cache_dir)
os.system('mkdir '+args.cache_dir)
def make_new_input_fn(filename=filename, nrepeat=10, batch_size=batch_size, buffer_size=buffer_size, cache_dir=None):

#    filename ='SDSS_cutouts/filename_0.02_0.065_Mstar_gt10_shuffledpixels_asinh.npy'
    names = np.loadtxt(filename, dtype=str)
   # names = np.load(filename)
    names = names[:67000]
    
    def input_fn():

        paths = tf.data.Dataset.from_tensor_slices(names)   
        dset = paths.map(loader, num_parallel_calls=batch_size)   
        dset = dset.flat_map(lambda arg: tf.data.Dataset.from_tensor_slices(arg))
        dset = dset.repeat(nrepeat)

        if cache_dir is not None:
            print('caching')
            dset = dset.cache(cache_dir)
        dset = dset.repeat().shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(batch_size)
        iterator = dset.make_one_shot_iterator()
        imgs_batch = iterator.get_next()
        return {'x':imgs_batch}
    
    def load(x):
        x = x.numpy()
        x = x.decode('utf-8')
        x = fits.getdata(x) +1.e-4   # for some reason need to use also the arcsinh(f-1000) normalization when using real data (not shuffledpixels)
        x = get_rotate(x)
        x = x[32:96,32:96]
        x = get_zoom(x) 
        return x

    def get_rotate(x):
        angle = np.random.uniform(0,180)
        x = rotate(x, angle, reshape=False)
        return x

    def get_zoom(x):
        x = zoom(x,0.5)
        return x

    def loader(y):
        imgs = tf.py_function(load,[y],[tf.float32])
        imgs = tf.cast(imgs, tf.float32)
        imgs = tf.expand_dims(imgs, axis=-1)
        return  imgs
    
    return input_fn


def make_input_fn(filename=filename, nrepeat=10, batch_size=batch_size, buffer_size=buffer_size, cache_dir=None):
	
#    names = np.loadtxt(filename, dtype=str) 
    names = np.load(filename)
    #np.random.shuffle(names)
#    names = names[:75000]
    names = names[:39000]
    def input_fn():

        paths = tf.data.Dataset.from_tensor_slices(names)   
        dset = paths.map(loader, num_parallel_calls=batch_size)   ############################################### this needs to be tested!!

        dset = dset.flat_map(lambda arg: tf.data.Dataset.from_tensor_slices(arg))
        dset = dset.repeat(nrepeat)
#### data augmentation goes here
 #       dset = paths.map(loader)
        if cache_dir is not None:
            print('caching')
            dset = dset.cache(cache_dir)
        dset = dset.repeat().shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(batch_size)
        iterator = dset.make_one_shot_iterator()
        imgs_batch = iterator.get_next()
        return {'x':imgs_batch}
        
    def decode_and_get_image(x):
      #  x = x.decode('utf-8')
#        x = x.strip('../')
        with fits.open(x) as f:
#            data = 0.001*(f[0].data-1000)
            data = np.arcsinh(f[0].data-1000)/100 +1.e-4
#            data = np.array(f[0].data)/100 +1.e-4  #make data as small as possible and add small bias
            #data = data/np.max(data) #minmax scaler
        return data#[8:40,8:40]

    def load(x):
        x = x.numpy()

        x = np.array(list(map(decode_and_get_image, x))) #image is 160*160
        x = np.array(list(map(get_rotate,x)))
        x = x[:,48:112,48:112]   #crop to 64x64
#        x = np.array(list(map(get_zoom,x))) #zoom to 32x32
  #      x = decode_and_get_image(x)
        return x#[:,8:40,8:40]

    def get_zoom(x):
        x = zoom(x,0.5)
        return x

    def get_rotate(x):
        angle = np.random.uniform(0,180)
        x = rotate(x, angle, reshape=False)
        return x

    def loader(y):
        imgs = tf.py_function(load,[y],[tf.float32])
        imgs = tf.cast(imgs, tf.float32)
        imgs = tf.expand_dims(imgs, axis=-1)
        return imgs#[0] # ToDo figure out why I need [0]
	
    return input_fn

def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images


def image_tile_summary(name, tensor, rows=8, cols=8):
    tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)
 
# define model


def pixelcnn_model_fn(features, labels,mode, params, config):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Extract input images
    x = features['x']
    print('define model architecture')
    model_opt = { 'nr_resnet': 3, 'nr_filters': 64, 'nr_logistic_mix': 1, 'resnet_nonlinearity': 'concat_elu', 'energy_distance': False}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Build the model
        def make_model_spec():
            input_layer = tf.placeholder(tf.float32, shape=[1,image_size, image_size,1])

#            if ema_on:
#                ema = tf.train.ExponentialMovingAverage(decay=decay)
  #              model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
 #               assign_ema = tf.group(tf.assign(var, ema
            model = tf.make_template('model', model_spec)
            out = model(input_layer, None, ema=None, dropout_p=0., **model_opt)
            out = tf.layers.dense(out, 2, activation=None) # project the output to only 2 values
            loc, scale = tf.split(out, num_or_size_splits=2,axis=-1)
            scale = tf.nn.softplus(scale) + 1e-4
            distribution = tfp.distributions.Independent( tfp.distributions.Normal(loc=loc, scale=scale))
            log_prob = - distribution.log_prob(input_layer)
            grads = tf.gradients(log_prob[0], input_layer)
            samples = distribution.sample()

            img_distribution = tfp.distributions.Normal(loc=loc, scale=scale)
            img_likelihood = img_distribution.log_prob(input_layer)

            hub.add_signature(inputs=input_layer,
                              outputs={'grads':grads[0], 'sample':samples,'log_prob':log_prob, 'loc':loc, 'scale':scale, 'likelihood_map':img_likelihood})
        
        spec = hub.create_module_spec(make_model_spec, drop_collections=['checkpoints'])
        pixelcnn = hub.Module(spec, name="pixelcnn_module")
            
        hub.register_module_for_export(pixelcnn, "pixelcnn_out")
        predictions =  pixelcnn(x, as_dict=True)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    
    # Build the model
    def make_model_spec():
        input_layer = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, n_channels])
        
        model = tf.make_template('model', model_spec)
        print('pixelcnn openAI')
        out = model(input_layer, None, ema=None, dropout_p=0.5, **model_opt)
        print('build output')
        out = tf.layers.dense(out, 2, activation=None) # project the output to only 2 values
        loc, scale = tf.split(out, num_or_size_splits=2,axis=-1)
        scale = tf.nn.softplus(scale) + 1e-4
        
        # Build Gaussian model for output value
        distribution = tfp.distributions.Independent( tfp.distributions.Normal(loc=loc, scale=scale))
        sample = distribution.sample()
        log_prob = distribution.log_prob(input_layer)  # get sum logprob for each image

        img_distribution = tfp.distributions.Normal(loc=loc, scale=scale)
        img_likelihood = img_distribution.log_prob(input_layer)

        hub.add_signature(inputs=input_layer,
                          outputs={'sample': sample, 'log_prob': log_prob, 'scale':scale, 'loc':loc, 'img_likelihood':img_likelihood})

    spec = hub.create_module_spec(make_model_spec, drop_collections=['checkpoints'])
    pixelcnn = hub.Module(spec, name="pixelcnn_module", trainable=True)

    print('make pixelCNN')
    output = pixelcnn(x, as_dict=True)

    loglikelihood = output['log_prob']
    sample = output['sample']
#    loc = output['loc']
#    scale = output['scale']
#    innerscale = scale[:,12:20,12:20]
#    innerloc = loc[:,12:20,12:20]
    img_likelihood = output['img_likelihood']

    tf.summary.scalar('loglikelihood', tf.reduce_mean(loglikelihood))  # the mean likelihood of the batch
    #tf.summary.histogram('loc',loc)
    #tf.summary.histogram('scale',tf.log(scale))
    #tf.summary.histogram('innerscale',tf.log(innerscale))
    #tf.summary.histogram('innerloc',  innerloc)

    image_tile_summary("image", tf.to_float(x[:16]), rows=4, cols=4)
    image_tile_summary("recon", tf.to_float(sample[:16]), rows=4, cols=4)
    image_tile_summary("diff", tf.to_float(x[:16])-tf.to_float(sample[:16]), rows=4, cols=4)

    loss = -tf.reduce_mean(loglikelihood)

    # Training of the model
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                          params["max_steps"])
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss , global_step=global_step)

    eval_metric_ops = {"loglikelihood": tf.metrics.mean(tf.reduce_mean(loglikelihood))}

    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "step": global_step}, every_n_iter=100)
    
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      training_hooks = [logging_hook])





print('Build input')
data_input = make_new_input_fn(args.filename, nrepeat=10, batch_size=args.batch_size, buffer_size=args.buffer_size, cache_dir=args.cache_dir)

# train model

params={'learning_rate':0.000003, 'max_steps':1000000}
print('Build estimator')
mymodel = tf.estimator.Estimator(model_fn=pixelcnn_model_fn,
                                params=params, model_dir='trained/0.02_0.08/asinh_SDSS_blobsLike')
#hooks = [tf_debug.LocalCLIDebugHook(ui_type='curses')]
print('Training')
mymodel.train(data_input, steps=500)#, hooks=hooks) #


# save model
exporter = hub.LatestModuleExporter("tf_hub",
        tf.estimator.export.build_raw_serving_input_receiver_fn(data_input()))

out = exporter.export(mymodel, 'trained/0.02_0.08/asinh_SDSS_blobsLike', mymodel.latest_checkpoint())
print(out)









             
             
