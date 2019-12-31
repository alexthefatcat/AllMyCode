# -*- coding: utf-8 -*-"""Created on Thu Apr 11 10:48:24 2019@author: milroa1"""
"""
##########################################################################
##########################################################################
   Keras Examples
##########################################################################   
"""
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
#try a basic net to predict it

##########################################################################
#                    Some Keras Code
##########################################################################
# advacnded keras
# state of the optimizer
# variational autoencoder
# dropout
# my more complex activation function
# reinforcement learning in keras
# recurrent
# weights inatillization
# weights update and gradients
# manual bacth normlization
# manually do SGD 
# DO a model where in each layer only part of gets updated by gan, both, autonecoder

# are all similar they use gradients
# Note self-attention, adversary attack, deep dream

#KerasPlus
# >adverseral
# >vae
# >clonenet
# >manualbackprop
##########################################################################
#          Coding in Basic Model
##########################################################################

net ={ 0 : [Input(shape=(12,))]}
net[   1   ] = [Dense( 20 , activation= 'relu')(net[0][-1])]
net[   2   ] = [Dense( 50 , activation= 'relu')(net[1][-1])]
net[   3   ] = [Dense( 100, activation= 'relu')(net[2][-1])]
net[   4   ] = [Dense( 87 , activation= 'relu')(net[3][-1])]
net[  -1   ] = net[4]
net['Order'] = [0, 1, 2, 3, 4] 
model = Model(inputs=net[0][-1], outputs=net[-1][-1])
model.compile(optimizer='adam', loss='categorical_crossentropy')
##########################################################################
#           General flow of neural network
##########################################################################
lr = 0.3 # LearningRate
model = BuildModel()
x, x_, Ytrue, y_ = x, Ypred, Ytrue, x_ignore

ForwardPassLayers  = list(range(0, len(model)))    
BackwardPassLayers = list(reversed(ForwardPassLayers))   
Forward            = [None for _ in ForwardPassLayers]
    
x_ = x
for layerno in ForwardPassLayers:
    Forward[layerno] = x_
    x_ = model[layerno](x_) 
# x_ here is Ypred    
y_ = loss(Ytrue,x_)

for layerno in BackwardPassLayers: #  BackProp
    y_ = dy_dx(model[layerno], Forward[layerno], y_)
    update(model[layerno], -lr*y_ ) 
##########################################################################
#           Backprop layer
##########################################################################    
# learn to do manual backprop    



##########################################################################
#           Backend how Grad and Function works
##########################################################################
import numpy as np
from keras import backend as K
i      = K.placeholder(shape=(4,), name="input")
square = K.square(i)
grad   = K.gradients([square], [i])
f      = K.function([i], [i, square] + grad)# i i**2, 2**i#grad is [grad]
ival   = np.full((4,), 3) 
print( f([ival]) )
# K.function() creates a function first input i, second list is tensors realted to the input
  
##########################################################################
#                    Shortcut
##########################################################################
def HighwayNetify(model,x,resnet=False):
    #of the model 0 and -1 should have same size
    dim = model.input_shape
    if not model.output_shape==dim:
        print("In and Out should be the same size")
    model_out       = model(x)     
    if resnet:
       Add()([x, model_out])
    gate            = Dense(dim, activation="sigmoid")(x)
    gated_model_out = Multiply()([gate, model_out])
    x_gated         = Multiply()([(1-gate), x])   
    return Add()([gated_model_out, x_gated])


residual=model(shortcut)
out = add([shortcut, residual])


##########################################################################
#                      Adversal_Trick
##########################################################################
""" Adversal one net learns to fool another one
    It inputs data into the net and backprops the answer it wants
    freezes the wieght of the other net and learns to fool it
"""

label = trainded_model(img)#5
desired_label = 6

distorted_img = advers_net(img)+img
label         = trainded_model(distorted_img)
minimaze(rms(label-desired_label) rms(distorted_img-img))
#modifies the input image
#backprop the desired label#but dont change that nets wieghts
# minimize difference between label,desired_label  AND distorted,img-img

##########################################################################
#                   Make Nets Adviserial  GAN                            #
##########################################################################
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



##########################################################################
#                  GAN
##########################################################################


def get_random_images(batch_size,imgs=imgs):   
    random_image_nos = np.random.randint(low=0,high=img.shape[0],size=batch_size)
    return imgs[random_image_nos]
def generator_noise(batch_size):
    return  np.random.normal(0,1, [batch_size, 100])
def stretch(n,batch_size):
    return np.full((batch_size), n)

output_true_batch  = stretch(1.0,batch_size)
output_false_batch = stretch(0.0,batch_size)
#  You have a generator and a disciminator
generator    , gen_dic = create_nn
discriminator, dis_dic = create_nn
gen_gan = discriminator(generator(gen_dic[0]))

adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
disc.compile(   loss='binary_crossentropy', optimizer=adam)
gen_gan.compile(loss='binary_crossentropy', optimizer=adam)

for n in nbatch:
    noise            = generator_noise(batch_size)
    generated_images = generator.predict(noise)
    real_imgs        = get_random_images(batch_size)
    
    #Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_imgs       , output_true_batch)
    d_loss_fake = discriminator.train_on_batch(generated_images, output_false_batch)
        
    # Train the generator, to try and fool the discriminto        
    discriminator.trainable = False
    gen_gan = gan.train_on_batch(noise, output_true_batch)
    discriminator.trainable = True   


##########################################################################
#                 Manual Backprop
##########################################################################

layers = list(model.layers)

#Predict
for layer in layers:
    #first one is input so ignore
    print(layer.input_shape,layer.output_shape)
    w,b = layer.get_weights()[0],layer.get_weights()[1]
    config = layer.get_config()

#Calculate loss

# Backprop
for layer in reversed(layers):
    #first one is input so ignore
    print(layer.input_shape,layer.output_shape)
    w,b = layer.get_weights()[0],layer.get_weights()[1]
    #learning rate and update weights
    config = layer.get_config()

##########################################################################
#                   From Imputation Work 30-04-19
##########################################################################

#KerasPlus
class NeuralNetworkFuncs:
    """ Functions directly related to keras
    
        >>    CreateKerasNet(layers,af_dict=None,model=False,Print=False,**kwargs)
        >>    NormalizeLatentLoss(latents)  # NotWorking or Used
        >>    AutoencoderPredict2DataFrame(model,df)
        >>    ImputationLossFunctionGenerator(Losses_df,Mask_NA)   
    
    """
    
    
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

    def GetModelShape(model,batch=True):    
        if batch:
           return [model.layers[0].input_shape, model.layers[-1].output_shape]
        return [model.layers[0].input_shape[-1], model.layers[-1].output_shape[-1]]
    
    def NormalizeLatentLoss(latents):
         """ The latents will have a mean=0, std =1
             NormalizeLatentLoss(encoder_net[-1])
             ##  NOT WORKING PROPERLY                          """
         lat_mean  = K.mean(latents, axis=0)
         lat_var   = K.mean(K.square(latents-lat_mean), axis=0)
         norm_mean = K.mean(K.abs(lat_mean))
         norm_lat  = K.mean(K.abs(lat_var-1))
         return norm_mean + norm_lat
    
    def AutoencoderPredict2DataFrame(model,df):
        "The Output has to have the same numner of columns,reall this only used for Autoencoders"
        data = model.predict(df)
        return pd.DataFrame(columns=df.columns, index=df.index, data=data)
    
    def ImputationLossFunctionGenerator(Losses_df,Mask_NA):
        """    losses should be a DataFrame with the loss for each of the output to be trained on
                NA_block matrix        """     
        KMask_NA    = K.variable( value=Mask_NA.values               )
        kCE_mask    = K.variable( value=Losses_df.loc["CE"  ].values )
        kMAE_mask   = K.variable( value=Losses_df.loc["MAE" ].values )    
        kMSE_mask   = K.variable( value=Losses_df.loc["MSE" ].values )
        #kNA_present = K.variable( value=Losses_df.loc["NA_present"].values )
    
        def Loss_Function(yTrue,yPred):
            ce  = K.binary_crossentropy(yTrue, yPred)
            mse = K.square(yTrue - yPred)
            mae = abs(yTrue - yPred)        
            combined = (kCE_mask * ce) + (kMSE_mask * mse) + (kMAE_mask * mae )
            not_block = K.dot(yTrue, KMask_NA )# 1s if         
            combined = (1. -not_block) * combined        
            return K.mean(combined,axis=-1)
        return Loss_Function
    






# Copying, Cloning Structure,Cloneing including weights, indexing

# org code indexing I thinks works this way
model2.layers[0].set_weights(model1.layers[0].get_weights())

def copylayerfromanotherlayer(layerfrom,layerto):
    layerto.set_weights(layerfrom.get_weights())

# copies structure not weights
BestModel = keras.models.clone_model(model.model)
#insert the weights into model
BestModel.set_weights(Model.get_weights())

##########################################################################
#                   Transfer Learning
##########################################################################

#You can use pop() on model.layers and then use model.layers[-1].output to create new layers.

#Example:

from keras.models import Model
from keras.layers import Dense,Flatten
from keras.applications import vgg16
from keras import backend as K

model = vgg16.VGG16(weights='imagenet', include_top=True)

model.layers.pop()  #<
model.layers.pop()  #<

new_layer = Dense(10, activation='softmax', name='my_dense')

out = new_layer(model.layers[-1].output)

model2 = Model(model.input, out)
#####################################################
# or 
model = vgg16.VGG16(weights='imagenet', include_top=True)
small_model = Model(model.inputs, model.layers[-2].outputs)

out = Dense(10, activation='softmax', name='my_dense')(small_model.output)
model2 =  Model(model.input, out)

##########################################################################
#                   face autoencoder
##########################################################################
data augemention
  rotate
  zoomin
  noise
  occluation
  blur
  bias image
  nonlinear warp
# l1 nomrlziation on the latent differnce between these 


3*512*512
general_autoencoder # also train on none face images
3*512*512
face_autoencoder    #  
4*512*512 # 4th being mask
mask = face_autoenocer[3]
mask>1 = 1
mask<0 = 0

gen_loss = rms((img, general_autoencoder)*(1-mask))/3
fac_loss = rms((img, face_autoencoder)*(mask))
mask_loss = (sum(mask))^0.5
#if face mask is known the rms of it can be used




# should I keep the lower part of the net the same between the general and the face
# drop layer where diffent layers are dropped out
# what happens if mutiple face exist in image



# dropout for the kernal
# differnce between autoencoders and gans
# -sharp not blurry images, -less depencdent on location, -rembers textures




##########################################################################
#                   gautoencoder                                        #
##########################################################################
decoder = create()
encoder = create()



#def MaskBackProp(exceptions=None,lossfunc="mse"):
#    if exceptions is None:
#        exceptions = []
#
#    def LossFunction(yTrue,yPred):
#        if   lossfunc=="mse":
#           out = K.square(yTrue - yPred)            
#        elif lossfunc=="mae":  
#           out = abs(yTrue - yPred)  
#        elif lossfunc=="ce":              
#           out = K.binary_crossentropy(yTrue, yPred)
#        for i in exceptions:       
#           out[i] = 0
#        return K.mean(combined,axis=-1)
#    return LossFunction
    
# in the latent the first two are reserverd
# 1) gan or vae loss
# 2) recombined ?
fake_lat = fake_latent_generator()
fake_img = decoder(fake_lat)
real_img = random_real_img()
gan_leatent_0_postion#adversial autoencoder ? ##
  fake_lat2 = encoder(fake_img)
  real_lat  = encoder(real_img)
real_img2_ae  = decoder(real_lat,(lat[1]=0))
real_img2_gan = decoder(real_lat,(lat[1]=1))
gan_leatent_1_postion
   real_img2_gan
   real_img
rms_autoencoder
  real_img2_ae
  real_img
  
#try these:
#    latent = lantent+noise(u=0,s=1)
#    loss_latent =sum(log(latent)) ? or latetn**2 ?








"""
##########################################################################
                Keras  Backend
##########################################################################
"""
#%%#################################################################################
"""          This loss function has been rearanged to work              """
def NewLoss1(y_true,y_pred):
    p=0
    for i in range(3074):
        if (y_pred[i+1]-y_pred[i])<0:
           p+=(y_true[i]-y_pred[i])**2
        elif (y_pred[i+1]-y_pred[i])>0:
           p+=(y_true[i]-y_pred[i])**2+(y_true[i]-y_pred[i])*(y_pred[i+1]-y_pred[i])**2
        else:
           p+=(y_true[i]-y_pred[i])**2+0.5*(y_true[i]-y_pred[i])*(y_pred[i+1]-y_pred[i])**2 
    return p

def NewLoss2(y_true,y_pred):
    p=0
    for i in range(3074):
        shift = y_pred[i+1]-y_pred[i]
        diff  = y_true[i  ]-y_pred[i]
        if shift<0:
           p+=diff**2
        elif shift>0:
           p+=diff**2 + diff*(shift)**2
        else:
           p+=(diff)**2+0.5*(diff)*(shift)**2 
    return p

def NewLoss3(y_true,y_pred):
    p=0
    shifts = y_pred[1:3075] - y_pred[:3074]
    diffs  = y_true -y_pred
    diffs2 = diffs**2

    for i in range(3074):
        shift = shifts[i]
        diff  = diffs[i]
        if shift<0:
           p+=diffs2[i]
        elif shift>0:
           p+=diffs2[i] +       diff*(shift)**2
        else:
           p+=diffs2[i]  +0.5*(diff)*(shift)**2 
    return p

def NewLoss4(y_true,y_pred):
    p=0
    shifts = y_pred[1:3075] - y_pred[:3074]
    diffs  = y_true -y_pred
    diffs2 = diffs**2
    rpart =  diffs*(shifts)**2

    for i in range(3074):
        shift = shifts[i]
        diff  = diffs[i]
        if shift<0:
           p+=diffs2[i]
        elif shift>0:
           p+=diffs2[i] + lpart[i]
        else:
           p+=diffs2[i]  +0.5*lpart[i]
    return p
##################################################################################
"""                     Example of Keras Backend                         """

def NewLoss(y_true, y_pred):

    true = y_true[:3074] 
    pred = y_pred[:3074]
    predShifted = y_pred[1:3075]

    diff        = true - pred
    diffShifted = predShifted - pred

    pLeftPart  = K.square(diff)
    pRightPart = diff * K.square(diffShifted)

    greater = K.cast(K.greater(diffShifted,0),K.floatx())
    equal = 0.5 * K.cast(K.equal(diffShifted, 0), K.floatx())
    mask = greater + equal

    return K.sum(pLeftPart + (mask*pRightPart))

#################################################################################



















#%%################################################################################


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)





     def customLoss(yTrue,yPred):
         # K.print_tensor(yTrue)
         # Masks_4_keras["tNA_mask"       ], Masks_4_keras["CE_or_MSE_mask"], Masks_4_keras["NA_present"    ] 

         #global ce,mse,combined,k,v,yTrue_,yPred_,block_mask,not_block
         #yTrue_,yPred_ = yTrue,yPred

         ce  = K.binary_crossentropy(yTrue, yPred)
         mse = K.square(yTrue - yPred)
         mse = abs(yTrue - yPred)
         
         combined = (Masks_4_keras["kCE_mask"] * ce) + (Masks_4_keras["kMSE_mask"] * mse)

         not_block = K.dot(yTrue, Masks_4_keras["kNA_mask"] )# 1s if 
         
         block_mask = 1.-not_block
         combined = block_mask * combined
    
         return K.mean(combined,axis=-1) #K.sum(K.log(yTrue) - K.log(yPred))




###### gradients


model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
model.fit(x, y, batch_size=10, epochs=1000, verbose=0)

gradients = K.gradients(model.output, model.input)              #Gradient of output wrt the input of the model (Tensor)
print(gradients)


#sess = tf.Session()
sess = K.get_session()
sess.run(tf.global_variables_initializer())

discriminator.trainable = False



def Dense_SelectedFreeze(size):#part trainiable
    "Dense_SelectedFreeze([LEFT,SHARED,RIGHT],activation)"
    from keras.layers import Concatenate
    from keras.layers.core import Lambda    
    # merge_two = concatenate([merge_one, third_input])
    # x2= Dense( 20 , activation= 'relu')(x)

def Dense2(*args,**kwargs):
    from keras.layers import Concatenate    
    arg1,*args = args # arg1 = [20,40,20] #[train on autoencoder, train on both, train on gan]
    if args ==[]:
        args=None
    print(arg1,">",args)
    def func(*args,**kwargs):#*args
        return  Concatenate()([Dense(arg1_) for arg1_ in arg1])
    return func


net={0 : Input(shape=(12,))}
net[   1   ] = Dense( 20 , activation= 'relu')(net[0])
net[   2   ] = Dense2( [10,30,10] , activation= 'relu')(net[1])
net[   3   ] = Dense( 100, activation= 'relu')(net[2])
net[   4   ] = Dense( 87 , activation= 'relu')(net[3])
net[  -1   ] = net[4]
net['Order'] = [0, 1, 2, 3, 4] 
model = Model(inputs=net[0], outputs=net[-1])


net={0 : [Input(shape=(12,))]}
net[   1   ] = [Dense( 20 , activation= 'relu')(net[0][-1])]
net[   2   ] = [Dense( 50 , activation= 'relu')(net[1][-1])]
net[   3   ] = [Dense( 100, activation= 'relu')(net[2][-1])]
net[   4   ] = [Dense( 87 , activation= 'relu')(net[3][-1])]
net[  -1   ] = net[4]
net['Order'] = [0, 1, 2, 3, 4] 
model = Model(inputs=net[0][-1], outputs=net[-1][-1])






NeuralNetworkFuncs.GetModelShape(model,batch=False)

Feed = Input(shape=(12,))



def Dense2(unitss=[10,30,10], activation= 'relu'):
    from keras.layers import Concatenate 
    from keras.layers.core import Lambda       
    #Concatenate()([Dense(units=units,activation=activation) for units in unitss])
    #return Lambda(lambda x: Concatenate()([Dense(units=units,activation=activation)(x) for units in unitss]))
    #return lambda x: Concatenate()([Dense(units=units,activation=activation)(x) for units in unitss])
    def func(x):
        insize = NeuralNetworkFuncs.GetModelShape(x,batch=False)[-1]
        xx = Input(shape=(insize,))
        sublayers = [Dense(units=units,activation=activation)(xx) for units in unitss]
        sublayers = [Model(inputs=xx, outputs=group) for group in groups]
        return Concatenate()(sublayers)
    return func

Feed2 = Dense2()(Feed)

net={0 : Input(shape=(12,))}
net[   1   ] = Dense( 20 , activation= 'relu')(net[0])
net[   2   ] = Dense( 10 , activation= 'relu')(net[1])
net[   3   ] = Dense( 100, activation= 'relu')(net[2])
net[   4   ] = Dense( 87 , activation= 'relu')(net[3])
net[  -1   ] = net[4]
net['Order'] = [0, 1, 2, 3, 4] 
model = Model(inputs=net[0], outputs=net[-1])

layer.output_shape or  layer.input_shape.







#### someone else using lambda
from keras.models import Model

#create dense layers and store their output tensors, they use the output of models 1 and to as input    
d1 = Dense(64, ....)(Model_1.output)   
d2 = Dense(64, ....)(Model_1.output)   
d3 = Dense(64, ....)(Model_2.output)   
d4 = Dense(64, ....)(Model_2.output)   

cross1 = Lambda(myFunc, output_shape=....)([d1,d4])
cross2 = Lambda(myFunc, output_shape=....)([d2,d3])

#I don't really know what kind of "merge" you want, so I used concatenate, there are Add, Multiply and others....
output = Concatenate()([cross1,cross2])

################################################################################################
#      weights := weights + alpha* gradient(cost)
model.compile(optimizer='adam', loss='categorical_crossentropy')

a = np.array(model.get_weights())         # save weights in a np.array of np.arrays
model.set_weights(a + 1)                  # add 1 to all weights in the neural network
b = np.array(model.get_weights())         # save weights a second time in a np.array of np.arrays
print(b - a)                              # print changes in weights
################################################################################################




def clip_norm(g, c, n):
    """Clip the gradient `g` if the L2 norm `n` exceeds `c`.
    # Arguments
        g: Tensor, the gradient tensor
        c: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        n: Tensor, actual norm of `g`.
    # Returns
        Tensor, the gradient clipped if required.
    """
    if c <= 0:  # if clipnorm == 0 no need to add ops to the graph
        return g

    # tf require using a special op to multiply IndexedSliced by scalar
    if K.backend() == 'tensorflow':
        condition = n >= c
        then_expression = tf.scalar_mul(c / n, g)
        else_expression = g

        # saving the shape to avoid converting sparse tensor to dense
        if isinstance(then_expression, tf.Tensor):
            g_shape = copy.copy(then_expression.get_shape())
        elif isinstance(then_expression, tf.IndexedSlices):
            g_shape = copy.copy(then_expression.dense_shape)
        if condition.dtype != tf.bool:
            condition = tf.cast(condition, 'bool')
        g = tf.cond(condition, lambda: then_expression, lambda: else_expression)
        if isinstance(then_expression, tf.Tensor):
            g.set_shape(g_shape)
        elif isinstance(then_expression, tf.IndexedSlices):
            g._dense_shape = g_shape
    else:
        g = K.switch(K.greater_equal(n, c), g * c / n, g)
    return g


class Optimizer(object):
    """Abstract optimizer base class.
    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.
    All Keras optimizers support the following keyword arguments:
        clipnorm: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        clipvalue: float >= 0. Gradients will be clipped
            when their absolute value exceeds this value.
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if None in grads:
            raise ValueError('An operation has `None` for gradient. '
                             'Please make sure that all of your ops have a '
                             'gradient defined (i.e. are differentiable). '
                             'Common ops without gradient: '
                             'K.argmax, K.round, K.eval.')
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def set_weights(self, weights):
        """Sets the weights of the optimizer, from Numpy arrays.
        Should only be called after computing the gradients
        (otherwise the optimizer has no weights).
        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the optimizer (i.e. it should match the
                output of `get_weights`).
        # Raises
            ValueError: in case of incompatible weight shapes.
        """
        params = self.weights
        if len(params) != len(weights):
            raise ValueError('Length of the specified weight list (' +
                             str(len(weights)) +
                             ') does not match the number of weights ' +
                             'of the optimizer (' + str(len(params)) + ')')
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Optimizer weight shape ' +
                                 str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        K.batch_set_value(weight_value_tuples)

    def get_weights(self):
        """Returns the current value of the weights of the optimizer.
        # Returns
            A list of numpy arrays.
        """
        return K.batch_get_value(self.weights)

    def get_config(self):
        config = {}
        if hasattr(self, 'clipnorm'):
            config['clipnorm'] = self.clipnorm
        if hasattr(self, 'clipvalue'):
            config['clipvalue'] = self.clipvalue
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)





class Adam(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](
           https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(Adam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


grad = K.gradients('mean_squared_error', autoencoder.inputs)

target = K.placeholder()
loss = K.sum(K.square(generator.outputs[0] - target))
grad = K.gradients(loss, generator.inputs[0])[0]
update_fn = K.function(generator.inputs + [target], [grad])


model.compile(optimizer='adam', loss='categorical_crossentropy')

a = np.array(model.get_weights())         # save weights in a np.array of np.arrays
model.set_weights(a + 1)                  # add 1 to all weights in the neural network


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.# removed
    """

    def __init__(self, lr=0.01, decay=0.,  **kwargs):
        super(SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        #grads = K.gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = - lr * g  # velocity
            self.updates.append(K.update(m, v)) # keras.backend.update(x, new_x)
            new_p = p + v
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
in model    
self.optimizer =     
    
    







class Optimizer(object):
    """Abstract optimizer base class.
    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.
    All Keras optimizers support the following keyword arguments:
        clipnorm: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        clipvalue: float >= 0. Gradients will be clipped
            when their absolute value exceeds this value.
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if None in grads:
            raise ValueError('An operation has `None` for gradient. '
                             'Please make sure that all of your ops have a '
                             'gradient defined (i.e. are differentiable). '
                             'Common ops without gradient: '
                             'K.argmax, K.round, K.eval.')
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def set_weights(self, weights):
        """Sets the weights of the optimizer, from Numpy arrays.
        Should only be called after computing the gradients
        (otherwise the optimizer has no weights).
        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the optimizer (i.e. it should match the
                output of `get_weights`).
        # Raises
            ValueError: in case of incompatible weight shapes.
        """
        params = self.weights
        if len(params) != len(weights):
            raise ValueError('Length of the specified weight list (' +
                             str(len(weights)) +
                             ') does not match the number of weights ' +
                             'of the optimizer (' + str(len(params)) + ')')
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Optimizer weight shape ' +
                                 str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        K.batch_set_value(weight_value_tuples)

    def get_weights(self):
        """Returns the current value of the weights of the optimizer.
        # Returns
            A list of numpy arrays.
        """
        return K.batch_get_value(self.weights)

    def get_config(self):
        config = {}
        if hasattr(self, 'clipnorm'):
            config['clipnorm'] = self.clipnorm
        if hasattr(self, 'clipvalue'):
            config['clipvalue'] = self.clipvalue
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
















##########################################################################################
#   manual backprop not on model but on placeholders using Adam and categorical_crossentropy
##########################################################################################
from keras.optimizers import Adam
from keras import backend as K
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
import numpy as np

# inputs and targets are placeholders
x     = K.placeholder(name="x", shape=(None, 28*28))
ytrue = K.placeholder(name="y", shape=(None, 10))

# model parameters are variables
W = K.variable(np.random.random((28*28,10)).astype(np.float32))
b = K.variable(np.random.random((10,)).astype(np.float32))
params = [W, b]

# single layer model: softmax(xW+b) 
ypred = K.softmax(K.dot(x,W)+b)

# categorical cross entropy loss
loss = K.mean(K.categorical_crossentropy(ytrue, ypred),axis=None)

# categorical accuracy
accuracy = categorical_accuracy(ytrue, ypred)

# Train function
opt = Adam()
updates = opt.get_updates(params, [], loss)
train = K.function([x, ytrue],[loss, accuracy],updates=updates)

# Train the network
((xtrain, ytrain),(xtest, ytest)) = mnist.load_data()
xtrain = xtrain.reshape((-1, 28*28)) # flatten input image
ytrain = to_categorical(ytrain, 10)
for epoch in range(500):
	loss, accuracy = train([xtrain, ytrain])
	print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, loss, accuracy))
##########################################################################################
#x     = K.placeholder(name="x", shape=(None, 87))
#ytrue = K.placeholder(name="y", shape=(None, 87))
#
#model = autoencoder
#x_ = census_df.iloc[[1],:].values
#ytrue_ = x
#   I think this should work
ypred = model.predict(x)
loss = K.mean(K.binary_crossentropy(ytrue, ypred),axis=None)
parms = model.get_weights()

opt = Adam()
updates = opt.get_updates(params, [], loss)
#updates used:
#   grads = K.gradients(loss, params)
train = K.function([x, ytrue],updates=updates)
for n in range(10):
    xtrain = x_
    ytrain = ytrue_
    train([xtrain, ytrain])
##########################################################################################
from keras import backend as K
K.categorical_crossentropy(ytrue, ypred)




grads = K.gradients(model.output, model.input)



def reverse_generator(generator, X_sample, y_sample, title):
    """Gradient descent to map images back to their latent vectors."""

    latent_vec = np.random.normal(size=(1, 100))

    # Function for figuring out how to bump the input.
    target = K.placeholder()
    loss = K.sum(K.square(generator.outputs[0] - target))
    grad = K.gradients(loss, generator.inputs[0])[0]
    update_fn = K.function(generator.inputs + [target], [grad])

    # Repeatedly apply the update rule.
    xs = []
    for i in range(60):
        print('%d: latent_vec mean=%f, std=%f'
              % (i, np.mean(latent_vec), np.std(latent_vec)))
        xs.append(generator.predict_on_batch([latent_vec, y_sample]))
        for _ in range(10):
            update_vec = update_fn([latent_vec, y_sample, X_sample])[0]
            latent_vec -= update_vec * update_rate

    # Plots the samples.
    xs = np.concatenate(xs, axis=0)
    plot_as_gif(xs, X_sample, title) 
##############################################################################
##############################################################################

def reverse_generator(generator, X_sample, y_sample, title):
    """Gradient descent to map images back to their latent vectors."""
    # Function for figuring out how to bump the input.
    target = K.placeholder()
    loss = K.sum(K.square(generator.outputs[0] - target))
    grad      = K.gradients(loss, generator.inputs[0])[0]
    update_fn = K.function(generator.inputs + [target], [grad])
################################################################
    latent_vec = np.random.normal(size=(1, 100))
    # Repeatedly apply the update rule.
    xs = []
    for i in range(60):
        print('{i}: latent_vec mean={np.mean(latent_vec)}, std={np.std(latent_vec)}')
        xs.append(generator.predict_on_batch([latent_vec, y_sample]))
        for _ in range(10):
            update_vec = update_fn([latent_vec, y_sample, X_sample])[0]
            latent_vec -= update_vec * update_rate

    # Plots the samples.
    xs = np.concatenate(xs, axis=0)
    plot_as_gif(xs, X_sample, title) 







###############################################################
#                  BackPropogation                             #
################################################################
target    = K.placeholder()
loss      = K.sum(K.square(model.outputs[0] - target))
grad      = K.gradients(loss, model.inputs[0])[0]
update_fn = K.function(model.inputs + [target], [grad])
################################################################
# ------------------ From another one -------------------------
#update_fn_ = K.function(model.inputs, [loss, grad])# This for fetch_loss_and_grads
#loss_value, grad_values = update_fn_(img)
#img += step * grad_values
################################################################
#            Example Net
################################################################
def add_noise(X,mean=0.3):
    noise = mean*np.random.randn(*X.shape)
    return noise+X
    
def predict_test(i=3,return_=False,model=None):
    if model==None:
        model = globals()["model"]
    plot_=False    
    if i == "plot":
        plot_=True
        i=[n for n in range(300)]    
    if not type(i) is list:
        i = [i]
    Ypred = model.predict(X[i,:]).flatten()#[0]#[0]
    Ytrue = Y[i]
    if plot_:
        plt.scatter(Ypred,Ytrue)
        plt.show()
        return None        
    print(f"Ypred:{Ypred}, Ytrue:{Ytrue}")
    if return_:
        return Ypred,Ytrue

import numpy as np
import matplotlib.pyplot as plt
X = np.random.rand(2000,10)
Y = (X[:,4] + X[:,6] - X[:,8])**2

net={0 : Input(shape=(10,))}
net[ 1 ] = Dense( 6 , activation= 'relu')(net[0])
net[ 2 ] = Dense( 8 , activation= 'relu')(net[1])
net[ 3 ] = Dense( 8 , activation= 'relu')(net[2])
net[ 4 ] = Dense( 1 , activation= None)(net[3])
net[-1 ] = net[4]
model = Model(inputs=net[0], outputs=net[-1])
model.compile(loss="mean_squared_error", optimizer = "adam")
#import pdb;pdb.set_trace()
predict_test()
predict_test("plot") 
model.fit(X, Y, batch_size = 25, epochs = 13)
predict_test()
predict_test("plot") 
model.fit(X, Y, batch_size = 5, epochs = 5)
predict_test()
predict_test("plot") 
model.fit(X, Y, batch_size = 4, epochs = 5)
predict_test()
predict_test("plot") 
model.fit(X, Y, batch_size = 2, epochs = 15)
predict_test("plot")    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def QuickCreateModel():
    netn={0 : Input(shape=(10,))}
    netn[ 1 ] = Dense( 8 , activation= 'relu')(netn[0])
    netn[ 2 ] = Dense( 1 , activation= 'relu')(netn[1])
    modeln = Model(inputs=netn[0], outputs=netn[2])
    modeln.compile(loss="mean_squared_error", optimizer = "sgd")
    return netn,modeln
net2,model2 = QuickCreateModel()
net3,model3 = QuickCreateModel()
model2.fit(X, Y, batch_size = 25, epochs = 13)
model2.fit(add_noise(X,1), Y, batch_size = 4, epochs = 5)
model2.fit(add_noise(X,0.5), Y, batch_size = 4, epochs = 5)
model2.fit(X, Y, batch_size = 4, epochs = 5)
predict_test("plot",model=model2)


for n in range(1000):
   X_, Y_= X[n], Y[[n]]


##>> copied

lr = 0.03 # LearningRate
#x, x_, Ytrue, y_ = x, Ypred, Ytrue, x_ignore

ModelDepth         = len(model.layers)
ForwardPassLayers  = list(range(0, ModelDepth-1))    
BackwardPassLayers = list(reversed(ForwardPassLayers))   
Forward            = [None for _ in ForwardPassLayers]
    
def GetLayerFunction(model,layer_get=2): 
    for i,layer in enumerate(model.layers,-1):
        if i==layer_get:
            print(layer.input_shape,layer.output_shape)
            return K.function([layer.input],[layer.output])  

target    = K.placeholder()
loss      = K.sum(K.square(model.outputs[0] - target))
grad      = K.gradients(loss, model.inputs[0])[0]
update_fn = K.function(model.inputs + [target], [grad])

import numpy as np
from keras import backend as K
i      = K.placeholder(shape=(4,), name="input")
square = K.square(i)
grad   = K.gradients([square], [i])
f      = K.function([i], [i, square] + grad)# i i**2, 2**i#grad is [grad]
ival   = np.full((4,), 3) 
print( f([ival]) )

##<< copied
####################################################################


target    = K.placeholder()
loss      = K.sum(K.square(model.outputs[0] - target))
grad      = K.gradients(loss, model.inputs[0])[0]
update_fn = K.function(model.inputs + [target], [grad])

p={}
X_,Ytrue_=predict_test(return_ = True)


   

p[0].predict(X_)



def get_intermidate_layer_func(model,layer_get=2): 
    for i,layer in enumerate(model.layers):
        if i==layer_get:
            print(layer.input_shape,layer.output_shape)
            return K.function([layer.input],[layer.output])   

func = get_intermidate_layer_func(model)
func([np.array([[1,2,3,4,5,6]])])[0]
#converts thing to variable k.eval()




def rolling_prediction(model,arg,return_layers=False):
    layer_output={}
    for i,layer in enumerate(model.layers[1:]):
       func = K.function([layer.input],[layer.output]) 
       arg  = func([arg])[0]
       layer_output[i] = arg
    if return_layers:
        return layer_output
    return arg


out = [np.array([1,2,3,4,5,6,7,8,9,10])]
Ytrue = 9
y1 = model.predict(out[0].reshape(1,-1))
y2 = rolling_prediction(model,out)
layers = rolling_prediction(model,out,True)
Ypred = y1
Ydiff = Ttrue - Ypred
Yerr = (Ydiff**2)/2
Yloss = Ydiff# is the differntial of this so used






lr = 0.03 # LearningRate
#x, x_, Ytrue, y_ = x, Ypred, Ytrue, x_ignore

ModelDepth         = len(model.layers)
ForwardPassLayers  = list(range(0, ModelDepth-1))    
BackwardPassLayers = list(reversed(ForwardPassLayers))   
Forward            = [None for _ in ForwardPassLayers]
    
def GetLayerFunction(model,layer_get=2): 
    for i,layer in enumerate(model.layers,-1):
        if i==layer_get:
            print(layer.input_shape,layer.output_shape)
            return K.function([layer.input],[layer.output])   



 


####################################################################################################

def Loss(Ytrue,Ypred):
    if type(Ytrue) in [float,int]:
        Ytrue = np.array([[Ytrue]])
    return Ytrue - Ypred


def blank_list():return [None for n in model.layers[1:] ]
model_outs  = blank_list()
model_funcs = [ K.function([layer.input],[layer.output]) for layer in model.layers[1:] ] 

x_ = [np.array([1,2,3,4,5,6,7,8,9,10])]
for layerno in ForwardPassLayers:
    x_ = model_funcs[layerno]([x_[0].reshape(1,-1)])
    model_outs[layerno] = x_
# x_ here is Ypred    
y_ = loss(Ytrue,x_)

loss__ = K.sum(k.abs(9 - model.layers[-1]))
grads = K.gradients(loss, model)



for layerno in BackwardPassLayers: #  BackProp
    #y_ = dy_dx(model[layerno], Forward[layerno], y_)
    #update(model[layerno], -lr*y_ ) 
    #y_ = model_funcs[layerno], 
    #get_wieghts
    model = model_funcs[layerno]
    loss = loss - model_outs[layerno] + model(model_outs[layerno] -1)
    grad_ = K.gradients(loss, model)
    model.layers[layerno+1] = model.layers[layerno+1] + grads[layerno]    
####################################################################################################    


loss      = K.sum(loss)
grad      = K.gradients(loss, model.inputs[0])[0]



func = get_intermidate_layer_func(model)
func([np.array([[1,2,3,4,5,6]])])[0]






x_ = out
for mod in model_funcs:
    x_ = mod(x_)
    
for layerno in BackwardPassLayers: #  BackProp
    y_ = 




x     = K.placeholder(name="x", shape=(None, 28*28))
ytrue = K.placeholder(name="y", shape=(None, 10))

# model parameters are variables
W = K.variable(1).astype(np.float32))
b = K.variable(2).astype(np.float32))
params = [W, b]

# single layer model: softmax(xW+b) 
ypred = K.dot(x,W)+b

# categorical cross entropy loss
loss = K.mean(K.categorical_crossentropy(ytrue, ypred),axis=None)

# Train function
opt = Adam()
updates = opt.get_updates(params, [], loss)
train = K.function([x, ytrue],[loss, accuracy],updates=updates)





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#  Convolutional Autoencoder in Keras


import os
#os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(1, 28, 28)) # 1ch=black&white, 28 x 28

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img) #nb_filter, nb_row, nb_col
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

print("shape of encoded", K.int_shape(encoded))

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)

# In original tutorial, border_mode='same' was used. 
# then the shape of 'decoded' will be 32 x 32, instead of 28 x 28
# x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x) 
x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x) 

x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 5, 5, activation='sigmoid', border_mode='same')(x)
print("shape of decoded", K.int_shape(decoded))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist 
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/255. # 0-1.に変換
x_test = x_test.astype('float32')/255. 

x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

print x_train.shape

from keras.callbacks import TensorBoard
autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=128, shuffle=True, validation_data=(x_test, x_test), verbose=1)

import matplotlib.pyplot as plt
%matplotlib inline

# utility function for showing images
def show_imgs(x_test, decoded_imgs=None, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            ax = plt.subplot(2, n, i+ 1 +n)
            plt.imshow(decoded_imgs[i].reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

decoded_imgs = autoencoder.predict(x_test)
print("input (upper row)\ndecoded (bottom row)")
show_imgs(x_test, decoded_imgs)

##Denoising autoencoder
# Add random noise before training!
noise_factor = 0.5 
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.) 
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

show_imgs(x_test_noisy)

autoencoder.fit(x_train_noisy, x_train, nb_epoch=100, batch_size=128, shuffle=True, validation_data=(x_test_noisy, x_test))

# denoising
print("denoising")
decoded_imgs = autoencoder.predict(x_test_noisy)
show_imgs(x_test_noisy, decoded_imgs)

# what if we feed the original noise-free test images?
print("\nof course, it works with original noise-less images")
decoded_imgs = autoencoder.predict(x_test)
show_imgs(x_test, decoded_imgs) # yes, it works well without noise!

print("Training history")
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(1, 2, 1)
plt.plot(autoencoder.history.history['loss'])
ax1.set_title('loss')
ax2 = fig.add_subplot(1, 2, 2)
plt.plot(autoencoder.history.history['val_loss'])
ax2.set_title('validation loss')










