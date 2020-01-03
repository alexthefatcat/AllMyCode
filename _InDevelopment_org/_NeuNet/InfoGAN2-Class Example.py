# -*- coding: utf-8 -*-"""Created on Mon Oct  1 11:57:02 2018@author: milroa1"""
from NeuNet2 import NeuNet
import numpy as np
import tensorflow as tf
import sys,os
print( os.getcwd())
sys.path.append(r'H:\AM\_NeuNe')
batcher  = NeuNet.model.mnist.batcher
safe_log = NeuNet.model.extra.safe_log
#%%#############################   Parameters  ######################################################################################
batch_size, no_it                                     =  32 , 80000
imgsz28_28, no_of_classes10 , no_of_noise_channels16  = (28,28) , 10 , 16
total_for_generator26                                 = no_of_classes10 + no_of_noise_channels16
neural_network_layers = {"gen"  :[total_for_generator26, 256, imgsz28_28 ], 
                         "dis"  :[imgsz28_28           , 128, 1          ],
                         "q_net":[imgsz28_28           , 128, 10         ]} # 784 in the image

#%%##################################################################################################################################
class GAN():
    def __init__(self,neural_network_layers):
        self.gen_layers, self.dis_layers, self.q_net_layers = neural_network_layers["gen"], neural_network_layers["dis"], neural_network_layers["q_net"]
        
        self.generator     = NeuNet.neural_network( self.gen_layers  , actf={-1:"sigm"}) # similar to the decoder
        self.discriminator = NeuNet.neural_network( self.dis_layers  , actf={-1:"sigm"}) 
        self.q_net         = NeuNet.neural_network( self.q_net_layers, actf={-1:"soft"}) 
        
        self.X, self.Z, self.c = NeuNet.model.create_placeholders([self.dis_layers[0], no_of_noise_channels16, no_of_classes10]) # X is img feed into qnet+discrimnator,Z,C into generator 
        
        self.sampler_Z, self.sampler_c = NeuNet.model.extra.random_sample(self.Z, mode="uniform") , NeuNet.model.extra.random_sample(self.c, mode="nomial" )
        
        self.G_sample    = self.generator(tf.concat(axis=1, values=[self.Z, self.c]))
        self.D_real      = self.discriminator(self.X)
        self.D_fake      = self.discriminator(self.G_sample)
        self.Q_c_given_x = self.q_net(self.G_sample)
        ##############################################################################
        self.D_loss, self.D_solver = NeuNet.train(  safe_log(self.D_real ) + safe_log(1 - self.D_fake) ,  self.discriminator.var_list())
        self.G_loss, self.G_solver = NeuNet.train(  safe_log(self.D_fake )                             ,  self.generator.var_list()    )
        self.Q_loss, self.Q_solver = NeuNet.train( -tf.reduce_sum(safe_log(self.Q_c_given_x )*self.c,1),  self.generator.var_list() + self.q_net.var_list()    )# equation is cross entropy
        
    def train(self,imgs_real,Z_noise,c_noise):
        
        _, D_loss_curr = sess.run([self.D_solver, self.D_loss], feed_dict={self.X: imgs_real, self.Z: Z_noise, self.c: c_noise})
        _, G_loss_curr = sess.run([self.G_solver, self.G_loss], feed_dict={self.Z: Z_noise  , self.c: c_noise})
        _              = sess.run([self.Q_solver             ], feed_dict={self.Z: Z_noise  , self.c: c_noise})
        
        return D_loss_curr, G_loss_curr

#%%#############################   Training    ###################################################################################### 
GAN1 = GAN(neural_network_layers)

def plot_and_save(it, sess, batchsz2plot = 20):
        Z_noise = GAN1.sampler_Z(batchsz2plot)   
        
        idx          = list(zip(*[[m,int(m//(batchsz2plot/no_of_classes10))]  for m in range(batchsz2plot) ]))
        c_noise      = np.zeros([batchsz2plot, no_of_classes10])
        c_noise[idx] = 1 
        
        samples = sess.run(GAN1.G_sample, feed_dict = {GAN1.Z: Z_noise, GAN1.c: c_noise})
        NeuNet.model.mnist.plot_save(samples, int(it))

print("Start Training ...")
(x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new") #mnist = NeuNet.model.mnist.create()  
            
with NeuNet.model.runsession() as sess:
    for it in range(no_it):
        imgs_real, labels = batcher(x_train, y_train,batchsize=batch_size)
        Z_noise, c_noise  = GAN1.sampler_Z(), GAN1.sampler_c()
        D_loss_curr, G_loss_curr = GAN1.train.train(imgs_real, Z_noise, c_noise)

        if it % 1000 == 0:
            print(f'\nIter: {it},  D loss: {D_loss_curr:.4},  G_loss: {G_loss_curr:.4}')              
            plot_and_save(int(it/1000), sess)        


