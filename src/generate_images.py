import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import sys
import time
import argparse
plt.rcParams['figure.figsize']=(32,32)


parser = argparse.ArgumentParser()

parser.add_argument('-iter','--iteration',type=int,help='number of current iteration in a for loop where this script is called')
args = parser.parse_args()

#pixelcnn_model = hub.Module('trained/0.02_0.08/asinh_Mgt10_3resnets_rightOrder/800ksteps/pixelcnn_out')

pixelcnn_model = hub.Module('../trained/0.02_0.08/asinh_Mgt10_3resnets_rightOrder_shuffledpixels/1573724454/pixelcnn_out')
x = tf.placeholder(shape=(1,32,32,1), dtype=tf.float32)
d = pixelcnn_model(x,as_dict=True)
out = d['sample']
g = d['grads']
l = d['log_prob']

sess= tf.Session()
sess.run(tf.global_variables_initializer())

N = 3
k=0
l=0
fig, ax = plt.subplots(N,N, sharey=True, sharex=True)

N2 = N*N
tot_start = time.time()
for a in ax.ravel():
    start = time.time()
#    fig1,ax1 = plt.subplots(1,1)
    im = np.zeros((1,32,32,1))
    for i in range(32):
        for j in range(32):
            im[0,i,j,0] = sess.run(out, feed_dict={x: im})[0,i,j,0]

#    hdu = fits.PrimaryHDU(im[0,:,:,0])
#    hdul = fits.HDUList([hdu])
#    ran = np.random.randint(0, sys.maxsize)
#    hdul.writeto('./generated/0.02_0.08/asinh_Mgt10_3resnets_rightOrder/fits/gen_'+str(ran)+'.fits')
#    ax1.imshow(im[0,:,:,0])
#    fig1.colorbar()
#    fig1.savefig('./generated/0.02_0.08/asinh_Mgt10_3resnets_rightOrder/pngs/gen_'+str(ran)+'.png')
#    fig1.close()

    a.imshow(im[0,:,:,0])
    
    end = time.time()
    print('time elapsed:'+str(end-start))

tot_end = time.time()
print('time elapsed:'+str(tot_end-tot_start))

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('../generated/shuffledpixels_Mgt10_3resnets_rightOrder_'+str(args.iteration)+'.pdf')
