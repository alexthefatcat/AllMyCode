# -*- coding: utf-8 -*-"""Created on Mon Nov 19 09:42:53 2018@author: milroa1"""

from NeuNet2 import NeuNet
import tensorflow as tf
import numpy as np
batcher = NeuNet.model.mnist.batcher
VAE=NeuNet.model.extra.VAE

VAE_basic                  = False
VAE_Orginal                = False
VAE_Competative_Clustering = False
VAE_levels_model           = True
VAE_levels_model_5_5___5_5 = False
OMNI_AE                    = False
Finding_the_best_gan_loss  = True

(x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new")
# seperates the data into high confidence and low confidence and train on the low confidence ones but make sure not to memorize
# compreses the latent so only caries certain amount of info



if VAE_basic:
        print("\nRunning .....  VAE_basic\n\n")
        #%%#############################   Parameters  ######################################################################################
        batch_size, no_it       =  64 ,  4000
        encoder_layers, decoder_layers  =  VAE.create_encoder_generator_layer_sizes(image_size=(28,28), middle=40, latent_size=8) # 784 in the image# [imgsz28_28, 40, (2,n_latent26) ],[n_latent26, 40,   imgsz28_28   ] 
        #%%#############################   Actual VAE  ######################################################################################
        
        if "VAE1" not in dir():
            VAE1 = VAE(decoder_layers, encoder_layers)
        VAE2_std = VAE(decoder_layers, encoder_layers)
        
        #%%#############################   Training    ###################################################################################### 
        (x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new") #mnist = NeuNet.model.mnist.create()  
        
        with NeuNet.model.runsession() as sess:
            for it in range(no_it):
                imgs, labels = batcher(x_train, y_train,batchsize=batch_size)
                #sess.run(VAE1.optimizer, feed_dict = {VAE1.X_in: imgs, VAE1.Y: imgs }) 
                
                imgs2,_  = sess.run([VAE1.OUT,VAE1.optimizer], feed_dict = {VAE1.X_in: imgs, VAE1.Y: imgs }) 
                imgs_std = np.abs(imgs-imgs2)
                sess.run(VAE2_std.optimizer, feed_dict = {VAE2_std.X_in: imgs, VAE2_std.Y: imgs_std }) 
            
                if it % 1000 == 0: 
                    imgs_std_out = sess.run(VAE2_std.OUT, feed_dict = {VAE2_std.X_in: imgs}) 
                    
                    ls, imgs_out, i_ls, d_ls, mu, sigm = sess.run([VAE1.loss, VAE1.OUT, VAE1.img_loss, VAE1.latent_loss, VAE1.z_mean, VAE1.z_log_sigma_sq], feed_dict = {VAE1.X_in: imgs, VAE1.Y: imgs}) #latent_loss#z_log_sigma_sq
                    print(f'Iter: {it},  total_ls: {ls:.4}, mean img_ls: {np.mean(i_ls):.4},  mean lat_ls: {np.mean(d_ls):.4}')
                    NeuNet.model.mnist.plot(imgs[:12],imgs_out[:12],imgs_std_out[:12])
                    
            del it,ls, imgs_out, i_ls, d_ls, mu, sigm                       
        #%%############################    Testing the decoder   ############################################################################       

        random_latent = VAE1.latent_random_gauss_sample(10)
        imgs          = sess.run(VAE1.OUT, feed_dict = {VAE1.z: random_latent})
        NeuNet.model.mnist.plot(imgs)
             
        
        # Lets Try one gettingthose images of latent
        # 3 level VAE
        # competative VAE
        # latent compresion and adding noise on the latents unevely or dropout
        #np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
if VAE_Orginal:
        print("\nRunning .....  VAE_Orginal\n\n")
        #%%#############################   Parameters  ######################################################################################
        batch_size, no_it               =  64 ,  40000
        encoder_layers, decoder_layers  =  VAE.create_encoder_generator_layer_sizes(image_size=(28,28), middle=40, latent_size=8) # 784 in the image# [imgsz28_28, 40, (2,n_latent26) ],[n_latent26, 40,   imgsz28_28   ] 
        #%%#############################   Actual VAE  ######################################################################################
        
        VAE1 = VAE(decoder_layers, encoder_layers)
        
        #%%#############################   Training    ###################################################################################### 
        (x_train, y_train), (x_test, y_test) = NeuNet.model.mnist.create(mode="new") #mnist = NeuNet.model.mnist.create()  
        
        with NeuNet.model.runsession() as sess:
            for it in range(no_it):
                imgs, labels = batcher(x_train, y_train,batchsize=batch_size)
                sess.run(VAE1.optimizer, feed_dict = {VAE1.X_in: imgs, VAE1.Y: imgs }) 
           
                if it % 1000 == 0:                    
                    ls, imgs_out, i_ls, d_ls, mu, sigm = sess.run([VAE1.loss, VAE1.OUT, VAE1.img_loss, VAE1.latent_loss, VAE1.z_mean, VAE1.z_log_sigma_sq], feed_dict = {VAE1.X_in: imgs, VAE1.Y: imgs}) #latent_loss#z_log_sigma_sq
                    print(f'Iter: {it},  total_ls: {ls:.4}, mean img_ls: {np.mean(i_ls):.4},  mean lat_ls: {np.mean(d_ls):.4}')
                    NeuNet.model.mnist.plot(imgs[:12], imgs_out[:12], imgs_std_out[:12] )
                    
            del it,ls, imgs_out, i_ls, d_ls, mu, sigm                       
        #%%############################    Testing the decoder   ############################################################################       

        random_latent = VAE1.latent_random_gauss_sample(10)
        imgs          = sess.run(VAE1.OUT, feed_dict = {VAE1.z: random_latent})
        NeuNet.model.mnist.plot(imgs)
             
        



if VAE_Competative_Clustering:
            print("\nRunning .....  VAE_Competative_Clustering\n\n")
            """
            I guess after this trying a loss function which looks
            
                     **                 
                    *  *                
                   *    *               
                         **              
                  *        ****          
                               ********* 
                 *                      
                                        
                *                       
            """
            n_latent26,      imgsz28_28     =  4 , (28,28) # 784 in the image
            decoder_layers , encoder_layers = [n_latent26, 20,   imgsz28_28   ] , [imgsz28_28, 20, (2,n_latent26) ]
            batch_size,      no_it          =  1 ,  50000
            
            VAE_ind = list(range(10))
            VAEs    = [VAE(decoder_layers, encoder_layers) for _ in VAE_ind]
            
            loss = VAE_ind.copy()
            
            with NeuNet.model.runsession() as sess:
                for it in range(no_it):
                    imgs, labels = batcher(x_train, y_train, batchsize=batch_size)
                    # find one with lowest error
                    for i in VAE_ind:
                           loss[i]  = sess.run(VAEs[i].loss, feed_dict = {VAEs[i].X_in: imgs, VAEs[i].Y: imgs })  
  
                    VAE_2_bp = VAE_ind if it <1000 else [min(enumerate(loss), key=lambda x:x[1])[0]]

                    #update the one with lowest error
                    for ii in VAE_2_bp:
                         sess.run(VAEs[ii].optimizer, feed_dict = {VAEs[ii].X_in: imgs, VAEs[ii].Y: imgs }) 
                         
                    if it % 1000 == 0:
                        print(it,VAE_2_bp)
                        
            randoms = [np.random.normal(0, 1, n_latent26) for _ in range(10)]
            for i in range(10):
                imgs  = sess.run(VAEs[i].OUT, feed_dict = {VAEs[i].z: randoms})
                NeuNet.model.mnist.plot(imgs)



if VAE_levels_model or VAE_levels_model_5_5___5_5:
    
         def get_random_block_3_3_for_0_layer(batch_size, small=3, large=28):
             """  a batch of images are input this selects a random region from it collects it, we use
             this to train the first layer of the 3*stacked VAE
             Use this to get a random blocks to train the first layer
             """
             from random import randint
             imgs = list(batcher(x_train,  batchsize=batch_size))[0]
             
             st_small = small//2
             en_small = small - st_small
             en_large = large - en_small
             center   = [( randint(st_small, en_large), randint(st_small, en_large)) for n in range(batch_size)]
             def indexer(loc):
                return  [loc+n for n in range(-st_small,en_small)]
                          
             imgs2    = [imgs[ np.ix_( [i],  indexer(loc_x ) , indexer( loc_y)  ) ]  for i, (loc_x,loc_y) in enumerate(center)]
             imgs2    = np.concatenate(imgs2)
             return imgs2
         
         def divide_img_up(img,box=[3,3]):
            """assumes first dim is one, but devides the image up returns a nested list with all boxs of the box size """
            def coord(m,n):
                return slice(m*n,(m+1)*n)
            sz = img.shape   
            if len(sz)==2:
                img=img.reshape(1,sz[0],sz[1],1)
            if len(sz)==3:
                img=img.reshape(sz[0],sz[1],sz[2],1)
            sz = img.shape 
            out=[]    
            for x_i in range(sz[1]//box[0]):
                out1=[]
                for y_i in range(sz[2]//box[1]):
                    print(x_i, y_i)
                    img_i = img[:,coord(x_i, box[0]),coord(y_i, box[1]),:]
                    out1.append(img_i)
                out.append(out1)        
            return out      


         def resize_img_tf(imgs,scale):
            """ so if imgs is say shape (64,28,28,1) and scale = [3,3]
            will return mean image sp in this case (64,27,27,1)
            imgs_mean = sess.run(resize_img_tf(imgs,[3,3])) 
            
            """
            if len(imgs.shape)==3:
                imgs = imgs.reshape(imgs.shape+(1,))   
            if type(scale) in [int,float]:    scale= [1,scale,scale,1]
            if len(scale)==2:                 scale= [1]+scale+[1]
            if len(scale)==3:                 scale=     scale+[1]
            
            cropped  = [m*(n//m) for m,n in zip(scale, imgs.shape)]
            imgs     = imgs[:cropped[0],:cropped[1],:cropped[2],:cropped[3]]/(scale[0]*scale[1])
            cropped2 = [n//m for m,n in zip(scale, imgs.shape)][1:3]
            return tf.image.resize_images(imgs, cropped2)








if VAE_levels_model:
        print("\nRunning .....  VAE_levels_model\n\n")
        
     
        
#############################################################################
##            Build the Neural Network      
        
        
        layers            = [ [5,4], [12,8], [22,16] ]
        batch_size, no_it =  64 , 400
    
        def create_layers( imgsz28_28, mid, n_latent26):
           return [n_latent26, mid,   imgsz28_28   ] , [imgsz28_28, mid, (2,n_latent26) ]    
        #img 28_28#make 27_27   #>1) 9_9 (2+(1,mean)) #>2) 3_3 (6+(1,mean))   #>3 1_1 (18+1) # this is the mega_latent
        
        VAEs       = [None for n in layers]
        
        for i, layer in enumerate(layers):
            start = (3,3) + (() if i==0 else (layers[i-1][1],))
            dec, enc = create_layers( start, layer[0], layer[1] ) 
            VAEs[i] = VAE( dec, enc )
            print(i,":",enc,dec,"#",start,layer)
        del i ,layer
       #good now how to train and concat output

######################################  Train the 0th layer       
         



            
            
        count=0
        with NeuNet.model.runsession() as sess:
            for batch_size,no_it in [(5,300),(10,400),(64,400)]:
                
                for it in range(no_it):
                    
                       count=count+1
                       imgs = get_random_block_3_3_for_0_layer(batch_size)
                       imgs=imgs.reshape(64,3,3,1)
                       imgs = x_test[:64,:,:].reshape(64,28,28,1)
                       
                       sess.run(VAEs[0].optimizer, feed_dict = {VAEs[0].X_in: imgs, VAEs[0].Y: imgs }) 
                       
                       if count%80==0:
                            print(count,batch_size)
                            imgs_out = sess.run(VAEs[0].OUT, feed_dict = {VAEs[0].X_in: imgs, VAEs[0].Y: imgs}) #latent_loss#z_log_sigma_sq
                            NeuNet.model.mnist.plot(imgs[:12],imgs_out[:12],other_size=(3,3))
                    




#tf.reduce_mean

if VAE_levels_model_5_5___5_5:
        print("\nRunning .....  VAE_levels_model\n\n")
    #        params     
    # 1(although 0 really) (1*1)# sum or mean of image block
    # 5              (5*1)#(1*1) or (1*5)#(1*1)
    # 10             (5*1)#(1*5)  which is the same as (1*5)#(5*1)
    # 25             (5*5)

        def img_crop(imgs):
            # as we are working_ with 28*28 crop them to 25*25
            if len(imgs.shape)==3:   return imgs[:,1:26,1:26  ]
            if len(imgs.shape)==4:   return imgs[:,1:26,1:26,:]
        
        imgs = list(batcher(x_train,  batchsize=64))[0]     
        imgs = img_crop(imgs)




    
    #randoms = [np.random.normal(0, 1, VAEs[0].decoder_layers[0]) for _ in range(10)]
    #imgs    = sess.run(VAEs[0].OUT, feed_dict = {VAEs[0].z: randoms})
    #NeuNet.model.mnist.plot(imgs,other_size=(3,3))
    
            
    
if OMNI_AE: 
    
        def reparameterization(z_mean_and_z_log_sigma_sq, mode="test"):
                z_mean , z_log_sigma_sq  =  z_mean_and_z_log_sigma_sq[:,0,:] , z_mean_and_z_log_sigma_sq[:,1,:] #apperent this should be sq
                if mode in ["test",0,None]:
                    epsilon              =  tf.zeros(tf.shape(z_log_sigma_sq)) 
                else :
                    epsilon              =  tf.random_normal(tf.shape(z_log_sigma_sq)) 
                z                        =  z_mean + (epsilon * tf.exp(z_log_sigma_sq/2)) #so need to find sqroot
                return z, z_mean, z_log_sigma_sq
            
        def seperate_latent(lat):
           latp1 = lat[:, 1, [0,1,2]]#[Noise, is_it_vae, critic_is_it_real]
           latp2 = lat[:, :,    3:  ]
           return latp1,latp2
       
        def VAE_encode_new_reparameterization(encoder_net, enc_in):   
             lat = encoder_net(enc_in)
             latp1,latp2 = seperate_latent(lat)
             z, z_mean, z_log_sigma_sq = NeuNet.model.extra.VAE.reparameterization(latp2,mode="train")  
             return latp1,z, z_mean, z_log_sigma_sq            
    
        encoder_layers, decoder_layers    = NeuNet.model.extra.VAE.create_encoder_generator_layer_sizes((28,28), 100, 10) 
        enc_in, enc_out, dec_in, dec_out  = NeuNet.model.create_placeholders({"enc_in":encoder_layers[0],"enc_out":encoder_layers[-1],"dec_in":decoder_layers[0], "dec_out":decoder_layers[-1]})
        encoder_net                       = NeuNet.neural_network(encoder_layers, actf={-1:"none"}) 
        decoder                           = NeuNet.neural_network(decoder_layers, actf={-1:"none"}) 

        #reparmitize   ##><## encoder_net(enc_in)
        latp1,z, z_mean, z_log_sigma_sq = VAE_encode_new_reparameterization(encoder_net, enc_in)   
        
    # n is noise
    # latp1 = [noise_level, is_it_vae, critic_is_it_real]   ,  latp2 is tradional latent,   lat = latp1+latp2 , LAT([a,b,c],n) = [a,b,c]+latp2+n# n being noise to the latp2  e2() = just gives you latp2

    # maybe latpl1# [noise_level, vae?, fake?]    #vae? is the image real but has been encoded and decoded,  
    
    # First Runs(tradional VAE)                      d(e(img+n)), img
    # so they roughly match the distributions 
    
    # Second Runs                                  e(d(e(img))),  e(img)
    # This is a bit more complex but this means that encoder and decoder should be more symteric
    
    # Third/FouthRun [n1,n2 play around with]   e2(d(e2(img)+n1)+n2),  e2(img) 
    # This means latent space is warped to real latents and image is warped to image space
    # its important to have some n1=0 and n2=0# so that it works with normal data

    #  Fifth gans

    # sixth also look cycle gan
    # vae(concat(latent,zeroes(5))) = lots of mean(gan(concat(latent,noise(5))))

    #  1)vae,2)make sure latent of decoded image is same as image,3)denoising vae, 4) not sure adding noise to latent is best way, 5) gan runs
    #  maybe make a new autoencoder that has mean 0 sd 1 and encourages mutual infomration instaead of vAE

    img, lat no noise,, nimg,nlat# have noise added,    nnimg, nnlat  just noise 

    def MOD_LAT(LAT, latp1_down, latp1_up=None, Noise_to_be_added_to_latp2=0):
        """      latpl1# [noise_level, vae?, fake?]           """
        def splitLAT(LAT):
            latp1 = LAT[:3]
            latp2 = LAT[3:]
            return latp1, latp2
        latp1, latp2 = splitLAT(LAT)
        backprop(latp1_down+LAT)
        if latp1_up is None:
            LATOUT = LAT
        else :
            LATOUT = latp1_up + LAT
        if Noise_to_be_added_to_latp2 != 0:
            LATOUT = LATOUT + Noise(Noise_to_be_added_to_latp2)
        return LATOUT
    
    
        # First Run VAE mode:        d(e(img+n)), img         latp1*=[0,0,0]  
         d(MOD_LAT[e(img) , [0,0,0],[0,0,0]]) - img #vae learn
         d(MOD_LAT[e(nimg), [n,0,0],[0,0,0]]) - img #vae learn to denoise
    
        # Second Runs                                  e(d(e(img))),  e(img)
        # This is a bit more complex but this means that encoder and decoder should be more symteric
     
        MOD_LAT[e(d(e(img))), [0,1,0],[0,0,0] ],  - e(img)   # we want the encoder to detect the vae   but   we want the decoder to fool the encoder
    
       # what happens when enocder wants    e(d(z)) =1    but decoder wants   e(d(z))  =0 in situation of fake   but in real    e(zz) = 0
       # what about  backpro through encoder than *-1 gradients applied to decoder, or backup prop 1,and 0 but only update the nets agording to the equivlent gradient
       # n is a neural net   out n(in)
       # [x,y] is calcaute the gradient
       # [x,y,"u"] "u" to update net
       # [x,grad]#using gradients from above
       # fimg,rimg,flabel(1),rlabel(0)
       # fimg = d(z)
       # e[rimg,rlabel,"u"]
       # e[fimg,flabel,"u"]
       # grad_fimg = e[fimg, rlabel]  
       # d[z,grad_fimg,"u"]
    
       # Third Run
       
        MOD_LAT[e(d(MOD_LAT[e(nimg),[n,1,0],[0,0,0] ] )), [0,1,0],[0,0,0] ],  - e(nimg) 
    
    





if Finding_the_best_gan_loss: 
    # Competing gans then swap detector and generator
    # ones which are good wieght more more those loss
    # neural network to learn the loss function
    pass

















