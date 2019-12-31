# -*- coding: utf-8 -*-"""Created on Mon Oct  1 11:57:02 2018@author: milroa1"""
from NeuNet2 import NeuNet
import numpy as np
import tensorflow as tf
import sys,os
print( os.getcwd())
sys.path.append(r'H:\AM\_NeuNe')
batcher, safe_log, GAN, pick, VAE = NeuNet.model.mnist.batcher, NeuNet.model.extra.safe_log, NeuNet.model.extra.GAN, NeuNet.model.extra.pick, NeuNet.model.extra.VAE


#MODE  =  [0]
MODE  = "INFOGAN"
MODE = pick(["INFOGAN","NORMAL","BASIC_CLASS","SUPER_CODER"],  "SUPER_CODER"  )


def plot_and_save(it, sess, batchsz2plot = 20):
        if MODE in ["BASIC_CLASS"]:
           Z_noise = gan.sampler_Z(batchsz2plot)
        else:
           Z_noise = sampler_Z(batchsz2plot)
        if   MODE in ["INFOGAN"]  :
            idx          = list(zip(*[[m,int(m//(batchsz2plot/no_of_classes10))]  for m in range(batchsz2plot) ]))
            c_noise      = np.zeros([batchsz2plot, no_of_classes10])
            c_noise[idx] = 1 
            samples = sess.run(G_sample, feed_dict = {Z: Z_noise, c: c_noise})
        elif MODE in ["BASIC_CLASS"]:
            samples = sess.run(gan.G_sample, feed_dict = {gan.Z: Z_noise})
        else :
            samples = sess.run(G_sample, feed_dict = {Z: Z_noise})
            
        NeuNet.model.mnist.plot_save(samples, int(it))

#%%#############################   Parameters  ######################################################################################
batch_size, no_it  =  32 , 40000
imgsz28_28, no_of_classes10 , no_of_noise_channels16  = (28,28) , 10 , 16
total_for_generator26                                 = no_of_classes10 + no_of_noise_channels16
neural_network_layers = {"gen"  :[total_for_generator26, 256, imgsz28_28 ],"dis"  :[imgsz28_28, 128, 1]} 

if MODE in [ "INFOGAN" ]:
    print("Running INFOGAN ... ")
    
    neural_network_layers["q_net"]=[imgsz28_28 , 128, 10 ]# 784 in the image
        
    #%%##################################################################################################################################
    
    generator     = NeuNet.neural_network( neural_network_layers["gen"  ], actf={-1:"sigm"}) # similar to the decoder
    discriminator = NeuNet.neural_network( neural_network_layers["dis"  ], actf={-1:"sigm"}) 
    q_net         = NeuNet.neural_network( neural_network_layers["q_net"], actf={-1:"soft"}) 
    
    X, Z, c = NeuNet.model.create_placeholders([imgsz28_28, no_of_noise_channels16, no_of_classes10]) # X is img feed into qnet+discrimnator,Z,C into generator 
       
    sampler_Z, sampler_c = NeuNet.model.extra.random_sample(Z, mode="normal") , NeuNet.model.extra.random_sample(c, mode="onehot" )
        
    G_sample    = generator(tf.concat(axis=1, values=[Z, c]))
    D_fake      = discriminator(G_sample)
    D_real      = discriminator(X)
    Q_c_given_x = q_net(G_sample)
    ##############################################################################
    D_loss, D_solver = NeuNet.train(  -safe_log(D_real ) - safe_log(1 - D_fake )  ,  discriminator.var_list())
    G_loss, G_solver = NeuNet.train(  -safe_log(D_fake )                          ,  generator.var_list()    )
    Q_loss, Q_solver = NeuNet.train(  tf.reduce_sum( -safe_log(Q_c_given_x )*c,1) ,  generator.var_list() + q_net.var_list()    )# equation is cross entropy

    #%%#############################   Training    ###################################################################################### 

    print(" -Start Training ...")
    (x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new") #mnist = NeuNet.model.mnist.create()  
                
    with NeuNet.model.runsession() as sess:
        for it in range(no_it):
            imgs_real, labels = batcher(x_train, y_train,batchsize=batch_size)
            Z_noise, c_noise  = sampler_Z(batch_size), sampler_c(batch_size)
            
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: imgs_real, Z: Z_noise, c: c_noise})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_noise  , c: c_noise})
            _              = sess.run([Q_solver        ], feed_dict={Z: Z_noise  , c: c_noise})
        
            if it % 1000 == 0:
                print(f'\nIter: {it},  D loss: {D_loss_curr:.4},  G_loss: {G_loss_curr:.4}')              
                plot_and_save(int(it/1000), sess)        
    
    
    Z_noise, c_noise = GAN.batch_4_varying_latent_and_class(sampler_Z, sampler_c)
    samples = sess.run(G_sample, feed_dict = {Z: Z_noise, c: c_noise})
    NeuNet.model.mnist.plot_save(samples, 99)


##def sampler_Z(batch_size=batch_size):#    return [np.random.uniform(-1., 1., no_of_noise_channels16     ) for _ in range(batch_size)]
##def sampler_c(batch_size=batch_size):#    return np.random.multinomial(1, 10*[0.1], size=batch_size )  
    

if MODE in [ "NORMAL", "GAN", "STANDARD" ]:
    print("Running STANDARD GAN ... ")

    #%%##################################################################################################################################
    
    neural_network_layers["gen"  ][0]=16
    generator     = NeuNet.neural_network( neural_network_layers["gen"  ], actf={-1:"sigm"}) # similar to the decoder
    discriminator = NeuNet.neural_network( neural_network_layers["dis"  ], actf={-1:"sigm"}) 
    
    X, Z = NeuNet.model.create_placeholders([imgsz28_28, no_of_noise_channels16]) # X is img feed into qnet+discrimnator,Z,C into generator 
    sampler_Z = NeuNet.model.extra.random_sample(Z, mode="normal") 
        
    G_sample    = generator(Z)
    D_fake      = discriminator(G_sample)
    D_real      = discriminator(X)
    
    D_loss, D_solver, G_loss, G_solver = GAN.loss(D_real, D_fake, generator.var_list(), discriminator.var_list())
    ##############################################################################
    print("Start Training ...")
    (x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new") #mnist = NeuNet.model.mnist.create()  
                
    with NeuNet.model.runsession() as sess:
        for it in range(no_it):
            imgs_real, labels = batcher(x_train, y_train,batchsize=batch_size)
            Z_noise  = sampler_Z(batch_size)
            
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: imgs_real, Z: Z_noise})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_noise  })
        
            if it % 1000 == 0:
                print(f'\nIter: {it},  D loss: {D_loss_curr:.4},  G_loss: {G_loss_curr:.4}')              
                plot_and_save(int(it/1000), sess)        
    
    
if MODE in [ "BASIC_CLASS" ]:
        class basic():
            """
            # create a basic GAN
        
            imgsz28_28, no_of_noise_channels16  = (28,28) ,  16    
            gan = GAN.basic({ "gen": [no_of_noise_channels16, 256, imgsz28_28 ], "dis": [imgsz28_28, 128, 1]} )
        
            print("Start Training ...")
            (x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new") #mnist = NeuNet.model.mnist.create()  
                        
            with NeuNet.model.runsession() as sess:
                for it in range(no_it):
                    imgs_real, labels = batcher(x_train, y_train,batchsize=batch_size)
                    gan.Z_noise  = gan.sampler_Z(batch_size),
                    
                    _, D_loss_curr = sess.run(gan.D_sess()[0], gan.D_sess()[1])
                    _, G_loss_curr = sess.run(gan.G_sess()[0], gan.G_sess()[1])
                
                    if it % 1000 == 0:
                        print(f'\nIter: {it},  D loss: {D_loss_curr:.4},  G_loss: {G_loss_curr:.4}')              
                        plot_and_save(int(it/1000), sess)        
            """
            
            def __init__(self,neural_network_layers):
                if neural_network_layers["dis"][0] !=neural_network_layers["gen" ][-1]:
                    print("Error the first layer in the dsicminator should be equal to the last layer in the generator")
                    
                self.neural_network_layers = neural_network_layers
                
                self.generator     = NeuNet.neural_network( neural_network_layers["gen"  ], actf={-1:"sigm"}) # similar to the decoder
                self.discriminator = NeuNet.neural_network( neural_network_layers["dis"  ], actf={-1:"sigm"}) 
                
                self.X, self.Z = NeuNet.model.create_placeholders([neural_network_layers["dis"][0],neural_network_layers["gen" ][0]]) # X is img feed into qnet+discrimnator,Z,C into generator 
                self.sampler_Z = NeuNet.model.extra.random_sample(self.Z, mode="normal") 
                    
                self.create_D_fake__D_real()
                
                self.D_loss, self.D_solver, self.G_loss, self.G_solver = GAN.loss(self.D_real, self.D_fake, self.generator.var_list(), self.discriminator.var_list())

            def create_D_fake__D_real(self):
                self.G_sample    = self.generator(self.Z)
                self.D_fake      = self.discriminator( self.G_sample)
                self.D_real      = self.discriminator( self.X)  
                
            def D_sess(self):
                return [self.D_solver,self.D_loss  ]
            def G_sess(self):
                return [self.G_solver, self.G_loss ]
            def train(self,sess, imgs_real, Z_noise):
                
                 _, self.D_loss_curr = sess.run(self.D_sess(), {self.X: imgs_real, self.Z: Z_noise})
                 _, self.G_loss_curr = sess.run(self.G_sess(), {self.Z: Z_noise                  })
                 return  self.D_loss_curr ,  self.G_loss_curr               
#                _, D_loss_curr = sess.run(gan.D_sess(), {gan.X: imgs_real, gan.Z: Z_noise})
#                _, G_loss_curr = sess.run(gan.G_sess(), {gan.Z: Z_noise                  })   

        imgsz28_28, no_of_noise_channels16  = (28,28) ,  16    
        gan = basic({ "gen": [no_of_noise_channels16, 256, imgsz28_28 ], "dis": [imgsz28_28, 128, 1]} )
    
        print("Start Training ...")
        (x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new") #mnist = NeuNet.model.mnist.create()  
                    
        with NeuNet.model.runsession() as sess:
            for it in range(no_it):
                imgs_real, labels = batcher(x_train, y_train,batchsize=batch_size)
                Z_noise = gan.sampler_Z(batch_size)
                D_loss_curr, G_loss_curr = gan.train(sess, imgs_real, Z_noise )
            
                if it % 1000 == 0:
                    print(f'\nIter: {it},  D loss: {D_loss_curr:.4},  G_loss: {G_loss_curr:.4}')              
                    plot_and_save(int(it/1000), sess)   
    
####################################################################################################################################################################################################    

    
    



if MODE in [ "SUPER_CODER" ]:

        NN_layers = VAE.create_encoder_generator_layer_sizes(image_size=(28,28), middle=40, latent_size=8, dic=True)

        Encoder_net = NeuNet.neural_network( NN_layers["Enc"], actf={-1: "none"}) # similar to the decoder
        Decoder     = NeuNet.neural_network( NN_layers["Dec"], actf={-1: "sigm" }) 
        Enc_in, Enc_out, Dec_in, Dec_out  = NeuNet.model.create_placeholders([ NN_layers["Enc"][0], NN_layers["Enc"][-1], NN_layers["Dec"][0], NN_layers["Dec"][-1] ])

        def latent_seperator(latent,reverse=False):
            """     pre_latent = Encoder_net(x)
                    latent_GAN, pre_latent_VAE = latent_seperator(pre_latent)
                    
                    latent_VAE, z_mean, z_log_sigma_sq = VAE.reparameterization(pre_latent_VAE, "train")
                    latent = latent_seperator([latent_GAN, latent_VAE], reverse = True)       """     
            if reverse:
                return tf.concat(axis=1, values=[latent[0], latent[1]] )
            else:
                return latent[:,0,:2] , latent[:,:,2:]# latent_GAN[0,0,0] gan or vae mode, latent_GAN[0,1,0] real or fake
            
        def Encoder(x, mode="VAE"):
            pre_latent = Encoder_net(x)
            latent_GAN, pre_latent_VAE = latent_seperator(pre_latent)

            if   mode=="VAE":
                latent_VAE, z_mean, z_log_sigma_sq = VAE.reparameterization(pre_latent_VAE, "train")
            else:
                latent_VAE = VAE.reparameterization(pre_latent_VAE, "test")[0]
      
            if   mode=="VAE": 
                return  latent_seperator([latent_GAN, latent_VAE], reverse = True), z_mean, z_log_sigma_sq  
            else:
                return  latent_seperator([latent_GAN, latent_VAE], reverse = True)

        
        latent , z_mean, z_log_sigma_sq = Encoder(Enc_in, "VAE")
        OUT = Decoder(latent)
        
        VAE_latent_loss = VAE.basic_latent_loss( z_log_sigma_sq, z_mean )
        VAE_img_loss    = VAE.basic_img_loss(      OUT,         Dec_out,    mode="cross_entropy" )
        VAE_loss, VAE_optimizer = NeuNet.train( tf.reduce_mean( VAE_latent_loss) + tf.reduce_mean(VAE_img_loss),  Encoder_net.var_list() + Decoder.var_list())
    
        def VAE_session_train(sess, imgs):
            sess.run(VAE_optimizer, feed_dict = {Enc_in: imgs, Dec_out: imgs }) 

        def latent_least_squares(latent):
             OUT = Decoder(latent)
             
            
################################################################################################################
        (x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new")
        with NeuNet.model.runsession() as sess:
            for it in range(no_it):
                imgs, labels = batcher(x_train, y_train, batchsize=batch_size)
                VAE_session_train(sess, imgs)
                    
    # First Runs(tradional VAE)                      d(e(img+n)), img
    # so they roughly match the distributions 
    
    # Second Runs                                  e(d(e(img))),  e(img)
    # This is a bit more complex but this means that encoder and decoder should be more symteric
    
    # Third/FouthRun [n1,n2 play around with]   e2(d(e2(img)+n1)+n2),  e2(img) 
    # This means latent space is warped to real latents and image is warped to image space
    # its important to have some n1=0 and n2=0# so that it works with normal data

    # img, lat no noise,, nimg,nlat# have noise added,    nnimg, nnlat  just noise 
"""
GL[0,1] = when feed into the encoder is always
GL = GAN_LATNET[0,1,2] = ( fake_latent-real_latent, fake_img-real-img ,real_img-vae_reconstructed )
z(0) minimize l squared
z(1) trick vae

rimg=real img , lat=latnet, flat= fake latent

1 ) VAE                                        d(e(rimg+n)), rimg

2a) symteric test                              e(d(e(rimg))),  e(rimg)  #GL[1] trained to be differnt
2b) symetric test latent                       e(d(flat)),     flat     # try both vae and gan

3 ) noise at differnt levels n sometimes = 0   e2(d(e2(rimg)+n1)+n2),  e2(rimg) 

4) GAN-VAE detection            "gan like": rimg, d(e(rimg)+[1]), d(flat+[1]), "vae like":   d(e(rimg)+[0]), d(flat+[0])       

5) GAN (fake latent)            "real": rimg          "fake": d(flat+[0])             
------------------------------------------------------------------------------------------------------------

img1 = fimg             # maybe not needed
lat1 = e(img1,z1)+n2    # maybe not needed
img3 = d(lat1)          # maybe not needed 
lat4 = e(img3)          # maybe not needed 
----------------------------------------
img2 = rimg+n1    

lat2 = e(img2,z2)+n3 
lat3 = flat

img4 = d(lat2,z=0)     # VAE reconstructed  
img5 = d(lat2,z=1)     # GAN reproduced image similar to the input one 
img6 = d(lat3,z=0)     # VAE blurry fake image
img7 = d(lat3,z=1)     # tradinatal gan produced images 

lat5 = e(img4)  # should be identical to lat2
lat6 = e(img5)  # should be identical to lat2
lat7 = e(img6)  # should be identical to lat3
lat8 = e(img7)  # should be identical to lat3
----------------------------------------
1 ) VAE_train  img4, img2
GAN_train  img5(z=1)  img2 #>latent         fake_latent-real_latent

real_img-vae_reconstructed 
4) GAN-VAE_detection  reallike(img5,img7,img2) vaelike(img4,img6)
    So maybe used:- 
            img2, img4

2-3 ) make reencoded latents similar to orginal  (lat2,[lat5,lat6]) , (lat3,[lat7,lat8]) # as well as adding noise on the way

    
What type of GAN to use:    
 -raw image                                vs everything else
 -raw fake image                           vs everything else
 -uses real image data i.e real latent     vs everything else
 I think the best way is making the system capable of producing real images so the raw image no fake image } tested against anything the coder produces but porbably fake latent stuff
 So:-
    raw image vs fake latent image    (tradional)
    or maybe at first
      reproduced real image vs fake latent image
    then:-
       raw image vs fake latent image    (tradional)
 
    
    
"""

