# -*- coding: utf-8 -*-"""Created on Mon Aug  6 13:38:11 2018@author: milroa1"""

#import numpy as np, pandas as pd

#import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.layers import fully_connected

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import numpy as np

"""
neural_network
train
model.mnist

metaclasses
return different type object before inputs

so maybe this class can create 4 type of object
1) neural network
    like a function input and output

2) trainer
    loss, optimizer

3) model
   which would consist of a neural network and a trainer
   with the option of fitting like keras and skikitlearn
   
4) enviroment or system ?
     where multiple neural networks can be trained
     
others)
    help code with tensorflow
    plot images
    mnist data
    
############################    
save model
expand neural network    
do my own backprop
dropping
multidemnsional
wieghts regulization
distance wieghting in the neural network
more advanced activation functions


batch_size, no_it  =  32 , 80000
mnist = NeuNet.model.mnist.create()   
     
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for it in range(no_it):
    plot_save_every_1000_iteration_infogan(it, sess)

    imgs_real, label, Z_noise, c_noise = get_noises_and_images(batch_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: imgs_real, Z: Z_noise, c: c_noise})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_noise  , c: c_noise})
    _              = sess.run([Q_solver        ], feed_dict={Z: Z_noise  , c: c_noise})

    if it % 1000 == 0:
        #print('Iter: {}'.format(it),'D loss: {:.4}'. format(D_loss_curr),'G_loss: {:.4}'.format(G_loss_curr),"\n")           
        print(f'Iter: {it},  D loss: {D_loss_curr:.4},  G_loss: {G_loss_curr:.4}\n')  

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
with tfsessionrun() as sess:
    for n in range(100000):
        imgs_real, label, Z_noise, c_noise = get_noises_and_images(batch_size)
    
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: imgs_real, Z: Z_noise, c: c_noise})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_noise  , c: c_noise})
        _              = sess.run([Q_solver        ], feed_dict={Z: Z_noise  , c: c_noise})
        if it % 1000 == 0:
            plot_save_every_1000_iteration_infogan(it, sess)
            print(f'Iter: {it},  D loss: {D_loss_curr:.4},  G_loss: {G_loss_curr:.4}\n') 
    


def clip_gradients(cost)
     optimizer    = tf.train.AdamOptimizer(1e-3)
     grads, varis = zip(*optimizer.compute_gradients(loss))
     grads        = [ None if grad is None else tf.clip_by_norm(grad, -1., 1.) for grad in grads ]
     optimize     = optimizer.apply_gradients(zip(grads, varis))
     
and saving a neural net
dropnet, resnet , reinforment (recurrent net-LSTM)





"""


class NeuNet:
    
    class neural_network:
        """
        The idea is create a simple neural network class that will incoperates in others
        
        add extend add layers and delete
        __org , .
        """
        def __init__(self,layers, actf="relu"):
            self.__org = { "layers":layers, "actf":actf }
            self._setup_basics(actf, layers)
            self.paramaters  =  self._paramaters_class(layers)# this creates the wieghts and the baises which are protected when copying
            self.in_and_outs = {}
    
        def _setup_basics(self, actf, layers):
            """
            This can be used to recaluate properties if they change
            """
            self.actf    = actf
            self.layers  = layers
            self.nlayers = len(layers)
            self.shape   = [layers[ 0], layers[-1]] 
            
            #self.no_paramters = sum([(l1+1)*l2  for l1,l2 in zip(layers,layers[1:])])
            if   type(actf) is str :
                self.actfun =[actf for n in range(self.nlayers-1)]
            elif type(actf) is dict :
                temp = ["relu" for n in range(self.nlayers-1)]
                for k,v in actf.items():
                    temp[k]=v
                self.actfun = temp
            else :
                self.actfun = actf
                
        def feedforward(self):# connect and set up activation functions
        
            self.in_and_outs[0] = self.feed
            for layer_n in range(self.nlayers-1):
                self.in_and_outs[ layer_n+1 ] = self.Act_f( self.in_and_outs[ layer_n ], self.paramaters.wieghts[layer_n], self.paramaters.biases[layer_n], fun=self.actfun[layer_n], layer_n )
            self.value_out = self.in_and_outs[ layer_n+1 ]
            self.value_mid = self.in_and_outs[len(self.in_and_outs)//2] 
            return self.value_out
            
        class _paramaters_class:
            def __init__(self, layers):
                self.wieghts  = NeuNet.neural_network._create_weights( layers ) 
                self.biases   = NeuNet.neural_network._create_biases(  layers )
                if "mrelu" in self.actfun:
                        def create_mrelu(shape, depth=3):
                            #shape = array.get_shape() 
                            mrelu = {"mult": tf.zeros(shape+[depth]) , "addi": tf.zeros(shape+[depth]) }
                            mrelu["mult"]  [ ...,0 ] = -1
                            for n in depth:
                               mrelu["addi"]  [ ...,n ] = n
                            return mrelu 
                        self.mrelu_wieghts = [ create_mrelu(layers[i]) if n =="mrelu" else None   for i,n in enumerate(self.actfun)]                    
                 

        
        #def __call__(self,feed, source=None):
        def __call__(self,feed): 
            """     acts like function      
                    pp = NeaNet ..
                    X = tf.placeholder(tf.float32, shape=[None, 784])
                    pp(X) # then acts like function   """
            self.feed = feed 
            return self.feedforward()            
    
        def _rand(shape,shape2=None):
           """
            def xavier_init(size):
                in_dim = size[0]
                xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
                return tf.random_normal(shape=size, stddev=xavier_stddev)
            
            X    = tf.placeholder(tf.float32, shape=[None, 784])
            D_W1 = tf.Variable(xavier_init([784, 128]))"""
           if shape2 is not None:
               n_parms_in=1
               for m in shape:
                   n_parms_in = m*n_parms_in  
               return tf.random_normal(shape=shape+shape2, stddev=1. / tf.sqrt(n_parms_in / 2.))                  
           #old way    
           if not type(shape) in (list,tuple): 
               shape=[shape]
           return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))    
    
        def _create_weights(layers_):
            # this should work with multidimnesional, and work with old
            def convert2list(arg):
                if type(arg) in [int,float,str,bool]:
                    arg=[arg]
                return list(arg) 
            def create_weight(a1,a2,dtype=tf.float32,mode="old"):
                if mode=="new":
                    return tf.Variable(NeuNet.neural_network._rand(convert2list(a1), convert2list(a2)), dtype=dtype)                  
                return(tf.Variable(NeuNet.neural_network._rand(convert2list(a1)+convert2list(a2)),dtype=dtype))  
            
#            def create_weight(a1,a2,dtype=tf.float32):    
#                return(tf.Variable(NeuNet.neural_network._rand([a1 ,a2 ]),dtype=dtype))
            
            nlayers_ = len(layers_)
            weights  = [ create_weight( layers_[ layer_n   ], layers_[ layer_n+1 ], mode="new") for layer_n in range( nlayers_-1) ]
            return weights   
        
        def _create_biases(layers_):  
            def create_bias(a1,dtype=tf.float32):       
                return(tf.Variable(NeuNet.neural_network._rand(a1)))        
            nlayers_ = len(layers_)
            biases    = [ create_bias( layers_[ layer_n+1 ]) for layer_n in range( nlayers_-1) ]
            return biases


    
    
    
    
        def Act_f(self,X,W,B,fun="relu",layer_n=None):  
            """  OK THIS IS NOT GEARED UP FOR MULTIDIMENSIONAL ONES
            X=(256,256)
            W=(256,256,3,4,4)
            B=(3,4,4)
            out=(3,4,4)
            
            p2="ijklmnopq"[ndim(W)]
            p1,p3 = p2[:ndim(X)],p2[ndim(X):]
            np.einsum(f"{p1},{p2}->p3")
            
            Z=np.einsum('ijk,ijklmno->lmno',X,W)
            """
            def act_mrelu(net, mrelu):  
                """Check this works
                """
                net2 = mrelu["mult"]*(mrelu["addi"] + net)
                net2  = -tf.relu( net2 )
                out_1 = tf.math.reduce_sum(net2)
                out = net - tf.relu( out_1 )
                return out            
            
            
#            def matmul2org(A,B):
#                 p2="ijklmnopqrstu"[:B.get_shape().ndims]
#                 p1,p3 = p2[:A.get_shape().ndims],p2[A.get_shape().ndims:]
#                 #return f"{p1},{p2}->{p3}"
#                 print(f"a{p1},{p2}->a{p3}",A.get_shape(),B.get_shape())
#                 return tf.einsum(f"a{p1},a{p2}->a{p3}", X,W)

            def matmul2(A,B):
                 p2="ijklmnopqrstu"[:B.get_shape().ndims]
                 p1,p3 = p2[:A.get_shape().ndims-1],p2[A.get_shape().ndims-1:]
                 #return f"{p1},{p2}->{p3}"
                 ein_code = f"a{p1},{p2}->a{p3}"
                 #print(ein_code,A.get_shape(),B.get_shape())
                 return tf.einsum(ein_code, X,W)


            
            #net = tf.add(tf.matmul(X,W),B)
            part1=matmul2(X,W)                
            net = tf.add(part1, B)
#            M1,M2 = tf.Variable(tf.random_normal([64,26]))  ,  tf.Variable(tf.random_normal([26,256]))  
#            N  = tf.einsum('ijk,lk->ijl',M1,M2) 
#            N  = tf.einsum('ij,jk->ik',M1,M2) 
#    
           # print("X>",X.get_shape().ndims,X.get_shape(),"W>",W.get_shape().ndims,W.get_shape(),"net>",net.get_shape().ndims,net.get_shape())
            if   fun=="relu": out = tf.nn.relu( net ) 
            elif fun=="sigm": out = tf.nn.sigmoid( net )
            elif fun=="soft": out = tf.nn.softmax( net )
            elif fun=="none": out = net
            elif fun=="mrelu":
                out = act_mrelu(net, self.parameters.mrelu_wieghts[layer_n] )


            return  out #tf.add(tf.matmul(X,W)), B)     
        
        def var_list(self):
            return [  self.paramaters.wieghts , self.paramaters.biases ]
    
    ################################################################################################################################
    class model:

        class runsession(object):  
            """
            #sess = tf.Session()
            #sess.run(tf.global_variables_initializer())
            """
            def __init__(self):
                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())
            def __dir__(self):
                return dir(self.sess)
            def __getattr__(self, attr):
                 return getattr(self.sess, attr)
            def __setattr__(self, attr, val):
                if attr is "sess":
                   super().__setattr__(attr, val)
                else :
                   setattr(self.sess.attr, val)
            def __enter__(self):
                return self            
            def __exit__(self, type, value, traceback):
               pass   
           

        def create_placeholders(info,name=None):
            """ Is how to write multiple placeholders
                a,b,c = NeuNet.model.create_placeholders([5,(5,5),7])
                also if dict can be used where key is the name
                {"X":45,"Y":(16,16),"Z":40}            """
            if type(info) is dict:
                return [NeuNet.model.create_placeholders(vn, kname) for kname,vn in info.items()]
            
            if type(info) is list:
                return [NeuNet.model.create_placeholders(n) for n in info]            
            
            if type(info) in [int,tuple]:
                if type(info) is tuple:
                    shape=[None]+list(info) 
                else :
                    shape=[None]+[info] 
                if name is None:
                     return tf.placeholder(tf.float32, shape=shape)#into qnet and discriminator                    
                else:
                     return tf.placeholder(tf.float32, shape=shape, name=name)#into qnet and discriminator   

                
                




        class extra:
            
            def pick(l,val):
                if val is int:
                   if val<len(l):
                      return l[val]
                else:
                   if val in l:
                      return val
                raise Exception('My error!')            
            
            def safe_log( value,small_val=1e-8 ):
                return tf.log(value + small_val)
            def print_1000(i,str_in,ii=1000):
                if i%ii in [0] :
                    print(str_in)  
            def print_1000_gen_disc_loss(i,gl,dl,ii=1000):
                NeuNet.extra.print_1000(i,f'Step {i}: Generator Loss: {gl}, Discriminator Loss: {dl}',ii)
                   
            def noise(batch_size, latent_size):
                return NeuNet._rand([batch_size, latent_size]) 
            

            
            def random_sample(array,batch_no=None,min_v=-1.0,max_v=1.0,mode=["normal","uniform","onehot"][0],oldnew="old"):
                size = [d.value for d in array.get_shape()]
                if oldnew=="old":
                    if   mode=="onehot":
                        def fun(batch_sz):
                            size1=size[1]
                            return  np.random.multinomial( 1,size1*[1/size1], size=batch_sz )
                    elif mode=="uniform":
                        def fun(batch_sz):
                            size[0]=batch_sz
                            return np.random.uniform( min_v,max_v, size=size)                       
                    else :
                        def fun(batch_sz):
                            size[0]=batch_sz
                            return np.random.normal(    0,1, size=size)                     
                    return  fun               
                if oldnew=="new" :                              
                    def sample_f(*args):
                        if len(args)>0:
                            
                            size[0] = args
                        if   mode=="onehot":
                            return  tf.random.multinomial(min_v,max_v, size=size)
                        elif mode=="uniform":
                            return  tf.random.uniform(    min_v,max_v, size=size)  
                        else :
                            return  tf.random.normal(    min_v,max_v, size=size)                    
                    return sample_f            

            class GAN():
                def vary_each_latent(Z_noise, extra=0, values=[-1,1],first_unchanged=True):
                      """iterates through each of the latent variables and alters them allows
                      one to easily what each latent does
                      #batch size will 2*times the values if values are [-1,1]
                      as first pass latent_var[0]=-1, second pass  latent_var[0]= 1    
                      
                      Z_noise   = GAN.vary_each_latent(sampler_Z(1), n_classes+1, [-1.5,1.5])
                      
                      """
                      
                      n_values = len(values)
                      Z_noise = Z_noise + np.zeros([int(first_unchanged)+extra+(n_values*(Z_noise.shape[1])),1])#EXTRA 1 FOR ORGINAL
                      for i in range(Z_noise.shape[1]):
                          for i_value,value in enumerate(values):
                              Z_noise[int(first_unchanged)+i_value+(n_values*i), i]=value
                      return Z_noise
                  
                def nos_2_onehot(ints,size=10):
                    """this allows you to produce a batch easily with just numbers in a list
                    the size of the batch is the size of the list ,size=10 default
                    example
                    a=nos_2_onehot([1,1,1,1,6]) 
                    a.tolist()#>
                    
                    [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]              """
                    
                    if type(ints) is int:
                        ints=[ints]
                    out=np.zeros([len(ints),size])
                    for i,n in  enumerate(ints):
                        if n not in [None]:
                           if n<size:
                              out[i,n   ] = 1
                           else:
                              print(f"Error the number {n} which the {i}th one in the list is larger or equal to {size}")
                              out[i,size-1] = 1
                    return out
                
                def batch_4_varying_latent_and_class(sampler_Z, sampler_c):
                    """ Combines batch_4_varying_latent_and_class and nos_2_onehot to look at what all the classes do
                    good for seeing what the latent is doing 
                    example:
                        batch_4_varying_latent_and_class(sampler_c,sampler_Z)
                        
                    Z_noise, c_noise = batch_4_varying_latent_and_class(sampler_c,sampler_Z)
                    samples = sess.run(G_sample, feed_dict = {Z: Z_noise, c: c_noise})
                    NeuNet.model.mnist.plot_save(samples, 99)        
                        
                    """
                    n_classes = sampler_c(1).shape[1]
                    Z_noise   = NeuNet.model.GAN.vary_each_latent(sampler_Z(1), n_classes+1, [-1.5,1.5])
                    c_noise   = NeuNet.model.GAN.nos_2_onehot(  ([5]*(Z_noise.shape[0]-(n_classes+1))) + [None] + list(range(n_classes))    ,  n_classes)
                    return Z_noise, c_noise 
            
            
                def loss(D_real, D_fake, G_vars, D_vars, version="vanilla", logits=False):
                    """      Cross Entropy loss function
                        
                             cross_entropy = - sum(p *log (q)) 
                             
                             For Binary Classification
                             -  ( (y)*log((p)) + (1-y)*log((1-p)) 
                            
                             Discriminator
                            
                                 for fake(y=0) : - ( (0)*log((fake)) + (1-0)*log((1-fake)) = -log((1-fake)
                                 for real(y=1) : - ( (1)*log((real)) + (1-1)*log((1-real)) = -log((real)) 
                                
                                 Discriminator_Total_Loss = -log((real))  -log((1-fake)
                                        # we want to make discrimator output 1 for real data and 0 for fake,# only update Discrimator wieghts
                    
                             Generator
                                
                                  for fake(y=1 )=  - ( (1-0)*log((fake)) + (0)*log((1-fake)) 
                                           # y=1, this is what we want as we want to trick discimnator
                                           
                                  Generator_Total_Loss ==  - log(fake)
                                         # we want to make fake 1 # only update Generator_net, generator doesnt touch real
                              
                                
                            so:     D_loss, D_solver = NeuNet.train(  -safe_log(D_real ) - safe_log(1 - D_fake )  ,  D_vars)
                                    G_loss, G_solver = NeuNet.train(  -safe_log(D_fake )                          ,  G_vars)  
                
                
                                       G_sample = generator(z)
                                       D_real   = discriminator(X)
                                       D_fake   = discriminator(G_sample)                 
                                         
                                     D_loss, D_solver,G_loss, G_solver =  GAN_loss( D_real, generator.var_list(), G_vars, discriminator.var_list(), version="vanilla")
                          ###########################################################################################################################################           
                                     If a sigmoid function is used it may be advantagous to use the logit (net) instead of the  out and use option logits = True
                                     proof that sigmoid cross entropy is simpler
                                     
                           The logistic loss is    = x - x * z + log(1 + exp(-x))*help_1
                                                   =   - x * z + log(1 + exp(x))      # For x < 0, to avoid overflow in exp(-x), we reformulate the above
                                                   =  max(x, 0) - x * z + log(1 + exp(-abs(x)))
                                                    
                                    *help_1:  x + log(1 + exp(-x)) = log(exp(x)) + log(1 + exp(-x)) = log( exp(x)*1 + exp(x-x)) = log(1 + exp(x))               
                
                           """
                    def sig_cross_entrpy(logit, label):
                        if label ==0:
                            return  tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=tf.zeros_like(logit))
                        if label ==1:
                            return  tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=tf.ones_like( logit))   
                 
                    if version in ["vanilla","normal","norm","cross-entropy","infogan"]: 
                       if logits: 
                           D_loss, D_solver = NeuNet.train( sig_cross_entrpy(D_real, 1) + sig_cross_entrpy(D_fake, 0) ,  D_vars)
                           G_loss, G_solver = NeuNet.train( sig_cross_entrpy(D_fake, 1)                               ,  G_vars) 
                #           D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
                ##           D_loss_real = tf.reduce_mean(sig_cross_entrpy(D_logit_real, 1))           
                #           D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
                ##           D_loss_fake = tf.reduce_mean(sig_cross_entrpy(D_logit_fake, 0)))           
                #           D_loss      = D_loss_real + D_loss_fake
                ##           D_loss      = D_loss_real + D_loss_fake           
                #           G_loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
                ##           G_loss      = tf.reduce_mean(sig_cross_entrpy(D_logit_fake, 1)))         
                       else :       
                           D_loss, D_solver = NeuNet.train(  -NeuNet.model.extra.safe_log(D_real ) - NeuNet.model.extra.safe_log(1 - D_fake )  ,  D_vars)
                           G_loss, G_solver = NeuNet.train(  -NeuNet.model.extra.safe_log(D_fake )                          ,  G_vars) 
                           
                    if version in ["wasserstein_gan"]:        
                        pass
            #            D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
            #            G_loss = -tf.reduce_mean(D_fake)
            #            
            #            D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-D_loss, var_list=D_vars))
            #            G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=G_vars))
            #            clip_D   = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in D_vars] 
            #            
                    if version in ["least_square"]:         
                        
                        D_loss = 0.5 * (tf.reduce_mean((D_real - 1)**2) + tf.reduce_mean(D_fake**2))
                        G_loss = 0.5 *  tf.reduce_mean((D_fake - 1)**2)
                        
                        D_solver = (tf.train.AdamOptimizer(learning_rate=1e-4).minimize(D_loss, var_list=D_vars))
                        G_solver = (tf.train.AdamOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=G_vars))        
                
                    if version in ["wgan-gp"]: 
                       if logits: 
                           pass
            #    		# WGAN Loss
            #               D_loss = tf.reduce_mean( D_fake ) - tf.reduce_mean( D_real )
            #               G_loss = -tf.reduce_mean( D_fake )
            #        
            #        		# Gradient Penalty
            #               epsilon      = tf.random_uniform(shape=[batch_size,1,1,1],minval=0.,maxval=1.)
            #               X_hat        = X_real + epsilon * (X_fake - X_real)
            #               D_X_hat      = discriminator(X_hat)
            #               grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
            #               red_idx      = range(1, X_hat.shape.ndims)
            #               slopes       = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
            #               gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            #               D_loss = D_loss + 10.0 * gradient_penalty        
            #        
            #               gp_sum = tf.summary.scalar("Gradient_penalty", gradient_penalty)
            #        
            #               train_vars = tf.trainable_variables()
            #        
            #               for v in train_vars:
            #                   tf.add_to_collection("reg_loss", tf.nn.l2_loss(v))
            #        
            #               G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=beta1, beta2=beta2).minimize( G_loss, var_list=G_vars)
            #               D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=beta1, beta2=beta2).minimize( D_loss, var_list=D_vars)
            #    
                    return D_loss, D_solver, G_loss, G_solver 
                
            class VAE():
                ''' def __init__(self,decoder_layers,encoder_layers, gamma_100=100.0,capacity_25=25.0,block_b_vae = True):
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
                        return gamma * tf.abs(latent_loss - capacity)'''
                    
                def __init__(self,decoder_layers,encoder_layers, gamma_100=100.0,capacity_25=25.0,block_b_vae = True,mode="train"):
                    self.decoder_layers,self.encoder_layers,self.gamma_100,self.capacity_25,self.block_b_vae,self.mode = decoder_layers,encoder_layers,gamma_100,capacity_25,block_b_vae,mode
                    
                    self.X_in, self.Y = NeuNet.model.create_placeholders( {"X_in":encoder_layers[ 0], "Y":decoder_layers[-1]} )
                
                    self.encoder_net = NeuNet.neural_network(encoder_layers, actf={-1:"none"})#encoder has twice the paramaenters use reparimtize        
                    self.decoder     = NeuNet.neural_network(decoder_layers, actf={-1:"sigm"}) 
                    self.encoder     = lambda x: NeuNet.model.extra.VAE.reparameterization(self.encoder_net(x),self.mode)
                    
                    self.z, self.z_mean, self.z_log_sigma_sq = self.encoder(self.X_in)
                    self.OUT                                 = self.decoder( self.z  ) # Z LATENT
                    
                    #Kullback Leibler divergence: 
                        
                    self.latent_loss = self.basic_latent_loss( self.z_log_sigma_sq, self.z_mean )
                    self.img_loss    = self.basic_img_loss(    self.Y,              self.OUT,    mode="cross_entropy" )
                    self.latent_loss = tf.reduce_mean( NeuNet.model.extra.VAE.disentangle(self.latent_loss,gamma_100,capacity_25,block_b_vae))# B-VAE #https://github.com/miyosuda/disentangled_vae/blob/master/model.py
                    self.img_loss    = tf.reduce_mean( self.img_loss)
                    self.loss, self.optimizer = NeuNet.train(self.img_loss + self.latent_loss,  self.decoder.var_list() + self.encoder_net.var_list())
                    if mode == "test":
                        self.mse = tf.reduce_mean( tf.squared_difference( self.Y, self.OUT), [1,2])
                        self.mae = tf.reduce_mean( abs( self.Y - self.OUT)                 , [1,2])

                def latent_random_gauss_sample(self,batch_size=10):
                     return [np.random.normal(0, 1, self.decoder_layers[0]) for _ in range(batch_size)]

                def basic_latent_loss(z_log_sigma_sq, z_mean):
                    return -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq - tf.square(z_mean) - tf.exp( z_log_sigma_sq), 1)#REDUCE SUMS THEM TO ELIMINETATE AXIS
                
                def basic_img_loss(Y,OUT,mode="cross_entropy"):
                    if mode =="cross_entropy":
                         return -tf.reduce_sum(  NeuNet.model.extra.VAE.binary_cross_entropy( Y, OUT), [1,2])
                    if mode =="least_squares": 
                         return  tf.reduce_sum( tf.squared_difference( Y, OUT), [1,2])
                    
                def binary_cross_entropy(A,B):
                    """  binary_cross_entropy(Y, Y_pred) """
                    return ((A) * NeuNet.model.extra.safe_log( B ))    +    ((1-A) * NeuNet.model.extra.safe_log( 1-B ))
                
                def reparameterization(z_mean_and_z_log_sigma_sq, mode="test"):
                        z_mean , z_log_sigma_sq  =  z_mean_and_z_log_sigma_sq[:,0,:] , z_mean_and_z_log_sigma_sq[:,1,:] #apperent this should be sq
                        if mode in ["test",0,None]:
                            print("test")
                            epsilon              =  tf.zeros(tf.shape(z_log_sigma_sq)) 
                        else :
                            epsilon              =  tf.random_normal(tf.shape(z_log_sigma_sq)) 
                        z                        =  z_mean + (epsilon * tf.exp(z_log_sigma_sq/2)) #so need to find sqroot
                        return z, z_mean, z_log_sigma_sq
                
                def disentangle(latent_loss,gamma,capacity,block=False):
                    if block:
                        return latent_loss
                    return gamma * tf.abs(latent_loss - capacity)
                
                
                def create_encoder_generator_layer_sizes(image_size,middle,latent_size,middle2=None,dic=False):
                    """  encoder_layers, decoder_layers  =  create_encoder_generator_layer_sizes(image_size, middle, latent_size)  """
                    if type(middle) is not list:
                        middle=[middle]
                    if middle2 is None:
                        middle2=middle
                    if dic:
                        return {"Enc":[image_size ]+[ middle ]+[ (2,latent_size) ] ,"Dec":[latent_size ]+[ middle2 ]+[ image_size ] }
                    return [image_size ]+[ middle ]+[ (2,latent_size) ] , [latent_size ]+[ middle2 ]+[ image_size ]     


    
        class mnist:
            folder = 'D:\\MNIST_data'
            def plot(*samples,other_size=(28,28)):
                """
                plot(*samples,other_size=(28,28))
                """
                if len(samples)>1:
                    for sample in samples:
                      NeuNet.model.mnist.plot(sample,other_size=other_size)
                else:
                    samples=samples[0]
                    n_samples=len(samples)
                    grid_y = int(n_samples**0.5)
                    grid_x = int(1+ ((n_samples-1)/grid_y))
                    
                    fig = plt.figure(figsize=(grid_y, grid_x))
                    gs = gridspec.GridSpec(grid_y, grid_x)
                    gs.update(wspace=0.05, hspace=0.05)
            
                    for i, sample in enumerate(samples):
                        ax = plt.subplot(gs[i])
                        plt.axis('off')
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_aspect('equal')
                        plt.imshow(sample.reshape(other_size), cmap='Greys_r')
                    return fig
        
            def plot_save(samples,i):
                    if i==0:
                        import os
                        if not os.path.exists(NeuNet.model.mnist.folder+'\\out'):
                            os.makedirs(NeuNet.mnist.model.folder+'\\out')
                    fig = NeuNet.model.mnist.plot(samples)
                    filepath = NeuNet.model.mnist.folder+f'\\out\\{str(i).zfill(3)}.png'
                    print( filepath )
                    plt.savefig( filepath, bbox_inches='tight')
                    plt.close(fig)
                    
            def create(mode="old"):
                """
                imgs, labels = mnist.train.next_batch(size)
                so if size is 100 output is
                784 * 100                               """
                if mode=="old":
                    from tensorflow.examples.tutorials.mnist import input_data
                    return input_data.read_data_sets(NeuNet.model.mnist.folder, one_hot=True)
                else :
                    return NeuNet.model.mnist.load_mnist(out="list")

            def load_mnist(out="dict",path="D:\\mnist.npz"): 
                """Downloaded from:- "https://s3.amazonaws.com/img-datasets/mnist.npz"  """
                mnist = tf.keras.datasets.mnist  
                (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
                x_train, x_test = x_train / 255.0, x_test / 255.0
                if out =="dict":
                    return {"train":{"x":x_train,"y":y_train},"test":{"x":x_test,"y":y_test}}
                else :
                    return ((x_train, y_train), (x_test, y_test)) 
            
            def batcher(*args, batchsize=30):
                batchmax=list(set(arg.shape[0] for arg in args))
                if len(batchmax)==1:
                   batchrand = [random.randint(0,batchmax[0]-1) for n in range(batchsize)]
                   return (arg[batchrand] for arg in args)
                print("Error,    first dimensions are of each of the inputs",batchmax)
            
            def reshape_flatten_or_img(imgin,reverse=False):   
                if reverse:
                    return np.reshape(imgin,[-1, 28, 28])
                return     np.reshape(imgin,[-1, 784]   )      #also line 55:  imgs_real=reshape_img(imgs_real)


        
        
    def train(equation, var_list,optimizer="Adam", loss="reduce_mean"):
           if loss == "reduce_mean":
              loss_     = tf.reduce_mean( equation )
           if optimizer == "Adam" : 
              solver = tf.train.AdamOptimizer().minimize(loss_, var_list = var_list)
           return(loss_, solver)        
        
        
#class trainer:        
        
        
    
    
    
    
#if False:        
#      import copy
#      _copy_obj      = copy.copy
#      _copy_obj_deep = copy.deepcopy       
#        
#      def unlinked_clone(self):
#          return NeuNet.copy_obj_deep(self)   
#      def linked_copy(self)
#         obj = NeuNet._copy_obj(self)
#         obj._paramaters_class._src = kwargs["source"]
#          return obj
        
  






#
#    def __dir__(self):
#        natdir = set(list(self.__dict__.keys()) + dir(self.__class__))
#        print(self.mana)
#        natdir.remove("domagic")
#        return list(natdir)
#    def domagic(self):
#        if self.mana <= 0:
#            raise "NotEnoughMana"
#        print("Abracadabra!")
#        self.mana -= 1





#    def __init__(self,layers,mode="neural-network", actf="relu"):
#        self.mode = mode
#        self._hide_dir=[]
#        
#        if mode=="neural-network":
#            self._hide_dir=["__self","__modify_object","model","neural_network","train"]
#            self.__self = NeuNet.neural_network(layers,actf)
#            self.__self.mode="neural-network"
#            NeuNet.__modify_object(self,"combine", self.__self)
#            
#    def __call__(self,call): 
#        """     acts like function      
#                pp = NeaNet ..
#                X = tf.placeholder(tf.float32, shape=[None, 784])
#                pp(X) # then acts like function   """
#        if self.mode=="neural-network":
#            self.feed = call 
#            return self.feedforward() 
#     
#    def __modify_object(self,mode,data):
#        if mode == "delete":
#            for n in data:
#                if n in self.__dict__:
#                    del self.__dict__[n]
#        if mode == "combine":
#            #def dir1(obj):return [x for x in dir(obj) if not x.startswith("_")]
#            #self.__dict__.update({k:v for k,v in data.__dict__.items() if k not in self.__dict__})
#            dirself = dir(self)
#            for attr in dir(data):
#                if not attr in dirself:
#                    setattr(self,attr,getattr(data,attr))
#                    
#        if mode == "move":
#            move_these = data[0]
#            move_to    = data[1]
#            self.__dict__[move_to]={}
#            for k in move_these:
#                self.__dict__[move_to][k]=self.__dict__[k]
#                del self.__dict__[k]
                
#    def __dir__(self):
##        all_dir = set(list(self.__dict__.keys()) + dir(self.__class__))
##        for attr in self._hide_dir+["_hide_dir"]:
##            if attr in all_dir:
##                all_dir.remove(attr)
##        return list(all_dir)
#         if self.mode=="neural-network":
#             return list(set(list(self._self.__dict__.keys()) + dir(self._self.__class__))) 
#         else :
#             return list(set(list(self.__dict__.keys()) + dir(self.__class__)))
#         
#    def __setattr__(self,attr,value):
#        print(attr)
#        if attr in ["mode"]:
#           #setattr(self,attr,value)
#           super().__setattr__(attr, value)
#            
#        if self.mode=="neural-network":
#            setattr(self._self,attr,value)
#        else :
#            #setattr(self,attr,value)
#            self[attr]=value
#            
#    def __getattr__(self,attr):
#        print(attr)
#        [1,2,3,4,5][random.choice([0,1,2,3,4,5])]
#        if self.mode=="neural-network":
#            if attr in dir(self._self):
#               return getattr(self._self,attr)
#            else :
#                raise Exception("Missing")
#                
    #####################################################################################################################
