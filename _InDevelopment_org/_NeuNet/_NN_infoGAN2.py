# -*- coding: utf-8 -*-"""Created on Mon Oct  1 11:57:02 2018@author: milroa1"""
from NeuNet2 import NeuNet
import numpy as np
import tensorflow as tf
import sys,os
print( os.getcwd())
sys.path.append(r'H:\AM\_NeuNe')
batcher = NeuNet.model.mnist.batcher

img_size784, number_of_classes10 , number_of_noise_channels16   =   (((28,28) , 10 , 16),(784 , 10 , 16))[1]
def reshape_img(imgin,reverse=False):    return  np.reshape(imgin,[-1]+  ([784],[28,28])[reverse]  )  #also line 55:  imgs_real=reshape_img(imgs_real)

total_for_generator26      = number_of_classes10 + number_of_noise_channels16

def plot_and_save(it, sess, batchsz2plot = 16):
        Z_noise = sample_Z(batchsz2plot, number_of_noise_channels16)
        idx = np.random.randint(0, number_of_classes10)
        c_noise = np.zeros([batchsz2plot, number_of_classes10])
        c_noise[range(batchsz2plot), idx] = 1        
        samples = sess.run(G_sample, feed_dict={Z: Z_noise, c: c_noise})
        NeuNet.model.mnist.plot_save(samples, int(it))

        
def sample_Z(m, n):    return np.random.uniform(-1., 1., size=[m, n])
def sample_c(m   ):    return np.random.multinomial(1, 10*[0.1], size=m)

def get_noises_and_images(batch_size):
    imgs, labels = batcher(x_train, y_train,batchsize=batch_size) #imgs, label = mnist.train.next_batch(batch_size) #label not used
    return imgs, labels, sample_Z(batch_size, number_of_noise_channels16), sample_c(batch_size)
        
generator     = NeuNet.neural_network([total_for_generator26, 256, img_size784 ], actf={-1:"sigm"}) # similar to the decoder
discriminator = NeuNet.neural_network([img_size784          , 128,    1        ], actf={-1:"sigm"}) 
q_net         = NeuNet.neural_network([img_size784          , 128,   10        ], actf={-1:"soft"}) 

X, Z, c = NeuNet.model.create_placeholders([img_size784, number_of_noise_channels16, number_of_classes10]) # X is img feed into qnet+discrimnator,Z,C into generator
   
G_sample    = generator(tf.concat(axis=1, values=[Z, c]))
D_real      = discriminator(X)
D_fake      = discriminator(G_sample)
Q_c_given_x = q_net(G_sample)
##############################################################################
D_loss, D_solver = NeuNet.train(  tf.log(D_real + 1e-8) + tf.log(1 - D_fake + 1e-8)  ,  discriminator.var_list())
G_loss, G_solver = NeuNet.train(  tf.log(D_fake + 1e-8)                              ,  generator.var_list()    )
Q_loss, Q_solver = NeuNet.train( -tf.reduce_sum(tf.log(Q_c_given_x + 1e-8)*c,1)      ,  generator.var_list() + q_net.var_list()    )# equation is cross entropy

print("Start Training ...")

batch_size, no_it  =  32 , 80000
(x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new") #mnist = NeuNet.model.mnist.create()  
            
with NeuNet.model.runsession() as sess:
    for it in range(no_it):
    
        imgs_real, labels, Z_noise, c_noise = get_noises_and_images(batch_size)
        imgs_real=reshape_img(imgs_real)###
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: imgs_real, Z: Z_noise, c: c_noise})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_noise  , c: c_noise})
        _              = sess.run([Q_solver        ], feed_dict={Z: Z_noise  , c: c_noise})
    
        if it % 1000 == 0:
            print(f'\nIter: {it},  D loss: {D_loss_curr:.4},  G_loss: {G_loss_curr:.4}')              
            plot_and_save(int(it/1000), sess)        





