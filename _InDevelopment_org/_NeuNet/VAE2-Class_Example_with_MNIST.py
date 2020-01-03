# -*- coding: utf-8 -*-"""Created on Tue Oct 30 12:16:12 2018@author: milroa1"""

"""                                 VAE - Variational AutoEnoder                                                 """

from NeuNet2 import NeuNet
import tensorflow as tf
import numpy as np
batcher = NeuNet.model.mnist.batcher
safe_log = NeuNet.model.extra.safe_log

#%%#############################   Parameters  ######################################################################################
batch_size, no_it       =  64 ,  4000
n_latent26, imgsz28_28  =  8 , (28,28) # 784 in the image
decoder_layers , encoder_layers= [n_latent26, 40,   imgsz28_28   ] , [imgsz28_28, 40, (2,n_latent26) ]

#%%#############################   Actual VAE  ######################################################################################
class VAE():
    def __init__(self,decoder_layers,encoder_layers, gamma_100=100.0,capacity_25=25.0,block_b_vae = True):
        self.decoder_layers,self.encoder_layers,self.gamma_100,self.capacity_25,self.block_b_vae = decoder_layers,encoder_layers,gamma_100,capacity_25,block_b_vae
        
        self.X_in, self.Y = NeuNet.model.create_placeholders( {"X_in":encoder_layers[ 0], "Y":decoder_layers[-1]} )

        self.encoder_net = NeuNet.neural_network(encoder_layers, actf={-1:"none"})#encoder has twice the paramaenters use reparimtixe        
        self.decoder     = NeuNet.neural_network(decoder_layers, actf={-1:"sigm"}) 
        self.encoder     = lambda x: VAE.reparameterization(self.encoder_net(x))
        
        self.z, self.z_mean, self.z_log_sigma_sq = self.encoder(self.X_in)
        self.OUT                                 = self.decoder( self.z  ) # Z LATENT
        
        #Kullback Leibler divergence: 
        self.latent_loss = -0.5 * tf.reduce_sum(1.0 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp( self.z_log_sigma_sq), 1)#REDUCE SUMS THEM TO ELIMINETATE AXIS
        self.img_loss    = [-tf.reduce_sum(  VAE.binary_cross_entropy( self.Y, self.OUT), [1,2]) , tf.reduce_sum( tf.squared_difference( self.Y, self.OUT), [1,2])  ][0]
        self.latent_loss = tf.reduce_mean( VAE.disentangle(self.latent_loss,gamma_100,capacity_25,block_b_vae))# B-VAE #https://github.com/miyosuda/disentangled_vae/blob/master/model.py
        self.img_loss    = tf.reduce_mean( self.img_loss)
        self.loss, self.optimizer = NeuNet.train(self.img_loss + self.latent_loss,  self.decoder.var_list() + self.encoder_net.var_list())
        
        def binary_cross_entropy(A,B):
            """  binary_cross_entropy(Y, Y_pred) """
            return ((A) * safe_log( B ))    +    ((1-A) * safe_log( 1-B ))
    
    def reparameterization(z_mean_and_z_log_sigma_sq, autoencode=False):
            z_mean , z_log_sigma_sq  =  z_mean_and_z_log_sigma_sq[:,0,:] , z_mean_and_z_log_sigma_sq[:,1,:] #apperent this should be sq
            epsilon                  =  tf.random_normal(tf.shape(z_log_sigma_sq)) 
            z                        =  z_mean + (epsilon * tf.exp(z_log_sigma_sq/2)) #so need to find sqroot
            return z, z_mean, z_log_sigma_sq
    
    def disentangle(latent_loss,gamma,capacity,block=False):
        if block:
            return latent_loss
        return gamma * tf.abs(latent_loss - capacity)

VAE1 = VAE(decoder_layers, encoder_layers)

#%%#############################   Training    ###################################################################################### 
(x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new") #mnist = NeuNet.model.mnist.create()  

with NeuNet.model.runsession() as sess:
    for it in range(no_it):
        imgs, labels = batcher(x_train, y_train,batchsize=batch_size)
        sess.run(VAE1.optimizer, feed_dict = {VAE1.X_in: imgs, VAE1.Y: imgs }) 
    
        if it % 1000 == 0:
            ls, imgs_out, i_ls, d_ls, mu, sigm = sess.run([VAE1.loss, VAE1.OUT, VAE1.img_loss, VAE1.latent_loss, VAE1.z_mean, VAE1.z_log_sigma_sq], feed_dict = {VAE1.X_in: imgs, VAE1.Y: imgs}) #latent_loss#z_log_sigma_sq
            print(f'\nIter: {it},  total_ls: {ls:.4}, mean img_ls: {np.mean(i_ls):.4},  mean lat_ls: {np.mean(d_ls):.4}')      
            NeuNet.model.mnist.plot(imgs[:12],imgs_out[:12])
    del it,ls, imgs_out, i_ls, d_ls, mu, sigm                       
#%%############################    Testing the decoder   ############################################################################       
randoms = [np.random.normal(0, 1, n_latent26) for _ in range(10)]
imgs    = sess.run(VAE1.OUT, feed_dict = {VAE1.z: randoms})
NeuNet.model.mnist.plot(imgs)
