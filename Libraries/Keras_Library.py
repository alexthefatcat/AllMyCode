# -*- coding: utf-8 -*-
"""Created on Tue Dec 31 19:25:09 2019@author: Alexm"""


def CreateKerasNet(layers,af_dict=None,model=False,Print=False,**kwargs):
    """     net = Create_Keras_Net([34,23,12])
            # Notes
               net[0 ] # input 
               net[-1] # output
               
               #>> LEARN THESE
               # clone    
               # target_model.set_weights(model.get_weights())     
               # concat split  
    """
    if af_dict is None:
        af_dict={}
    af_dict["default"] = af_dict.get("default","relu")
    Print = kwargs.get("print",Print)
    
    printout = {True : print, False : lambda *args,**kwargs: None}[Print]
    
    def if_not_keras_layer(arg):
        return "tensorflow" not in str(type(arg))

    net={ "Order" : [0]}

    printout(f"{'#'*100}\n###   NeuralNetworkFuncs.CreateKerasNet({layers}, af_dict={af_dict}, model={model}, Print={Print})   ###\n{'#'*100}")
    if  if_not_keras_layer(layers[0]):
        net[0] = Input(shape=(layers[0],))
        printout(f"net={{0 : Input(shape=({layers[0]},))}}")
    else:
        net[0] = layers[0]
        printout(f"net={{0 : net2[0]}}")        
        
    for i,layer_i in enumerate(layers[1:],1):
        if  if_not_keras_layer(layers[0]):
            activation = af_dict.get(i,af_dict["default"]) 
            activation = af_dict.get(i-len(layers),activation) 
            net[ i ] = Dense(layer_i,activation=activation)(net[i-1])
            activation_ = " "+str(activation)+" " if activation is None else "'"+activation+"'"
            printout(f"net[   {i}   ] = Dense( {layer_i: <3d}, activation={str(activation_).rjust(7)})(net[{i-1}])")
        else :
            net[ i ] = layer_i(net[i-1])
            printout(f"net[ {i}  ] = net2[**](net[{i-1}])")
        net["Order"] += [i]
            
    net[-1] = net[i]

    printout(f"net[  -1   ] = net[{i}]") 
    printout(f"net['Order'] = {net['Order']} ")
    if not model:
       return net
    model_ = Model(inputs=net[0], outputs=net[-1])
    printout(f"model = Model(inputs=net[0], outputs=net[-1])")  
    printout("#"*100,"\n")
    return net, model_

def adversial_learn_BCE(discriminator,generator):
    """ Its good as well for the disriminator to have a sigmoid function as the af
    BCE for Binary Cross Entropy
    args are the models to train
    A function that given two models returns a function that trains them.
    
    example:
        
        adversial_loss = adversial_learn_BCE(discriminator,deciever)
        
        for n in nbatch:
            fake_latent = generator_noise(batch_size)
            real_img    = get_random_images(batch_size)
            adversial_loss(fake_latent, real_img)    
    """
    def stretch(n,batch_size):
        return np.full((batch_size), n)
    
    gan  = lambda x: discriminator(generator(x)) #?
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # Binary Cross Entropy = H(p,q) = -SUM(p(x)*log(q(x)))
    disc.compile(   loss='binary_crossentropy', optimizer=adam)
    gen_gan.compile(loss='binary_crossentropy', optimizer=adam)

    def adversial_loss(fake_latent, real_img):
        batch_size         = real_img.shape[0]
        output_true_batch  = stretch(1.0,batch_size)
        output_false_batch = stretch(0.0,batch_size)

        fake_img = generator.predict(fake_latent)
        #Train the discriminator        
        d_loss_real = discriminator.train_on_batch(real_img, output_true_batch)
        d_loss_fake = discriminator.train_on_batch(fake_img, output_false_batch)
        # Train the generator, to try and fool the discriminto r 
        discriminator.trainable = False
        gan = gan.train_on_batch(fake_latent, output_true_batch)
        discriminator.trainable = True   


adversial_loss = adversial_learn_BCE(discriminator,generator)

for n in nbatch:
    fake_latent = generator_noise(batch_size)
    real_img    = get_random_images(batch_size)
    adversial_loss(fake_latent, real_img)

##########################################################################
#                 VAE
##########################################################################

def VarAutoEncoderify(encoder, decoder, return_all=False):
    """ 
    Given an encoder,decoder will return a VAE
    That trains like a VAE
    creates a encoder2# with an extra z_log_var
    VAE(X,Y,batch,epochs)#X=Y in autoencoders
    """
    VAE_inputs = encoder.layers[ 0]
    shape      = encoder.layers[-1]
    z_mean     = encoder.layers[-1].output     
    z_log_var  = Dense(shape)(encoder.layers[-2].output  )
    encoder2   = Model(encoder.layers[0],[z_mean,z_log_var])
        
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_log_var)))
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    encoder3    = Lambda(sampling)(encoder2) 
    VAE_outputs = decoder(encoder3(VAE_inputs))     
    VAE         = Model(VAE_inputs, VAE_outputs)
    kl_loss     = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    
    def VAE_Loss(Ytrue,Ypred):
        rec_loss = keras.losses.mean_squared_error(Ytrue, Ypred)
        return K.mean(rec_loss + kl_loss)
    
    VAE.compile(optimizer='rmsprop', loss = VAE_Loss) 
    if return_all:
        return VAE, encoder2, sampling, VAE_Loss
    return VAE








 