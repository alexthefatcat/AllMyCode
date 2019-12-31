# -*- coding: utf-8 -*-"""Created on Tue Oct 30 12:16:12 2018@author: milroa1"""

"""                                 VAE - Variational AutoEnoder                                                 """

from NeuNet2 import NeuNet
import tensorflow as tf
import numpy as np
batcher = NeuNet.model.mnist.batcher
safe_log = NeuNet.model.extra.safe_log
#%%#############################   Parameters  ######################################################################################
batch_size, no_it       =  64 ,  4000
n_latent26, imgsz28_28  =  8 , (28,28) 
neural_network_layers = {"decoder":[n_latent26, 40, imgsz28_28  ], "encoder":[imgsz28_28, 40, (2,n_latent26) ]} # 784 in the image
gamma_100 , capacity_25 = 100.0 , 25.0, block_b_vae = True
#%%#############################   Actual VAE  ######################################################################################
def binary_cross_entropy(A,B):
    """  binary_cross_entropy(Y, Y_pred) """
    return ((A) * safe_log( B ))    +    ((1-A) * safe_log( 1-B ))

def reparameterization(z_mean_and_z_log_sigma_sq, autoencode=False):
        z_mean , z_log_sigma_sq  =  z_mean_and_z_log_sigma_sq[:,0,:] , z_mean_and_z_log_sigma_sq[:,1,:] #apperent this should be sq
        epsilon                  =  tf.random_normal(tf.shape(z_log_sigma_sq)) 
        z                        =  z_mean + (epsilon * tf.exp(z_log_sigma_sq/2)) #so need to find sqroot
        return z, z_mean, z_log_sigma_sq

def disentangle(latent_loss,gamma=gamma_100,capacity=capacity_25,block=False):
    if block:
        return latent_loss
    return gamma * tf.abs(latent_loss - capacity)

X_in, Y = NeuNet.model.create_placeholders( {"X_in":imgsz28_28,  "Y":imgsz28_28} ) 
#X_in, Y = NeuNet.model.create_placeholders( {"X_in":neural_network_layers["encoder"][ 0], "Y":neural_network_layers["decoder"][-1]} )
encoder_net = NeuNet.neural_network(neural_network_layers["encoder"], actf={-1:"none"})#encoder has twice the paramaenters use reparimtixe        
decoder     = NeuNet.neural_network(neural_network_layers["decoder"], actf={-1:"sigm"}) 
encoder     = lambda x: reparameterization(encoder_net(x))

z, z_mean, z_log_sigma_sq = encoder(X_in)
VAE                       = decoder( z  ) # Z LATENT

#Kullback Leibler divergence: 
latent_loss = -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq - tf.square(z_mean) - tf.exp( z_log_sigma_sq), 1)#REDUCE SUMS THEM TO ELIMINETATE AXIS
img_loss    = [-tf.reduce_sum(  binary_cross_entropy( Y, VAE), [1,2]) , tf.reduce_sum( tf.squared_difference( Y, VAE), [1,2])  ][0]
latent_loss = tf.reduce_mean( disentangle(latent_loss,block=block_b_vae))# B-VAE #https://github.com/miyosuda/disentangled_vae/blob/master/model.py
img_loss    = tf.reduce_mean( img_loss)

loss, optimizer = NeuNet.train(img_loss + latent_loss,  decoder.var_list() + encoder_net.var_list())
#%%#############################   Training    ###################################################################################### 
(x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new") #mnist = NeuNet.model.mnist.create()  

with NeuNet.model.runsession() as sess:
    for it in range(no_it):
        imgs, labels = batcher(x_train, y_train,batchsize=batch_size)
        sess.run(optimizer, feed_dict = {X_in: imgs, Y: imgs }) 
    
        if it % 1000 == 0:
            ls, imgs_out, i_ls, d_ls, mu, sigm = sess.run([loss, VAE, img_loss, latent_loss, z_mean, z_log_sigma_sq], feed_dict = {X_in: imgs, Y: imgs}) #latent_loss#z_log_sigma_sq
            print(f'\nIter: {it},  total_ls: {ls:.4}, mean img_ls: {np.mean(i_ls):.4},  mean lat_ls: {np.mean(d_ls):.4}')      
            NeuNet.model.mnist.plot(imgs[:12],imgs_out[:12])
                               
#%%############################    Testing the decoder   ############################################################################       
randoms = [np.random.normal(0, 1, n_latent26) for _ in range(10)]
imgs    = sess.run(VAE, feed_dict = {z: randoms})
NeuNet.model.mnist.plot(imgs)
