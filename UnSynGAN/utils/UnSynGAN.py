"""
  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".
  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.
  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.
"""

import numpy as np
from functools import partial

import keras.backend as K
from keras.models import Model
from keras.layers import Input, LeakyReLU, Reshape
from keras.layers import Conv3D, Add, UpSampling3D, Activation, Flatten, Embedding, Multiply, Lambda
from keras.optimizers import Adam
from keras.initializers import lecun_normal
gen_initializer = lecun_normal() 

import sys
sys.path.insert(0, './utils')
from layers import wasserstein_loss, ReflectPadding3D, gradient_penalty_loss, ConditionalInstanceNormalization3D, charbonnier_loss, total_variation_loss_3D
from Adam_lr_mult import LR_Adam

def resnet_blocks(input_res, weight, kernel, name):

    in_res_1 = ReflectPadding3D(padding=1)(input_res)
    out_res_1 = Conv3D(kernel, 3, strides=1, kernel_initializer=gen_initializer, 
                       use_bias=False,
                       name=name+'_conv_a', 
                       data_format='channels_first')(in_res_1)
    out_res_1 = ConditionalInstanceNormalization3D(name=name+'_isnorm_a')([out_res_1, weight])
    out_res_1 = Activation('relu')(out_res_1)
    
    in_res_2 = ReflectPadding3D(padding=1)(out_res_1)
    out_res_2 = Conv3D(kernel, 3, strides=1, kernel_initializer=gen_initializer, 
                       use_bias=False,
                       name=name+'_conv_b', 
                       data_format='channels_first')(in_res_2)
    out_res_2 = ConditionalInstanceNormalization3D(name=name+'_isnorm_b')([out_res_2, weight])
    
    out_res = Add()([out_res_2,input_res])
    return out_res

class UnSynGAN(object):
    def __init__(self, ImageRow=128, ImageColumn=128, ImageDepth=128, 
                 FirstDiscriminatorKernel = 16, FirstGeneratorKernel = 8,
                 lamb_rec = 1000, lamb_adv = 1, lamb_gp = 10, lamb_tv = 0.001,
                 lr_DisModel = 0.0001, lr_GenModel = 0.0001):
        self.ImageRow = ImageRow
        self.ImageColumn = ImageColumn
        self.ImageDepth = ImageDepth
        self.D = None                   # discriminator
        self.G = None                   # generator
        self.DisModel = None            # discriminator model
        self.GenModel = None            # generator model
        self.DiscriminatorKernel = FirstDiscriminatorKernel
        self.GeneratorKernel = FirstGeneratorKernel
        self.lamb_adv = lamb_adv
        self.lamb_rec = lamb_rec
        self.lamb_gp = lamb_gp
        self.lamb_tv = lamb_tv
        self.lr_DisModel = lr_DisModel
        self.lr_GenModel = lr_GenModel
        
    def label_expand(self,x):
        # unsqueeze
        x_unsqueeze = K.expand_dims(K.expand_dims(K.expand_dims(x, -1), -1), -1)
        return K.tile(x_unsqueeze, (1, 1, self.ImageRow, self.ImageColumn, self.ImageDepth))
           
        
    def discriminator_block(self, name):
        """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
        the input is real or generated. Unlike normal GANs, the output is not sigmoid and does not represent a probability!
        Instead, the output should be as large and negative as possible for generated inputs and as large and positive
        as possible for real inputs.
        Note that the improved WGAN paper suggests that BatchNormalization should not be used in the discriminator."""
    
        # In:
        inputs = Input(shape=(1, self.ImageRow, self.ImageColumn, self.ImageDepth), name='dis_input')
        EmbeddedClass = Input(shape=(1,))
        
        # Input 128
        nclass = Lambda(self.label_expand)(EmbeddedClass)
        inputs_embed = Multiply()([nclass,inputs])
        
        disnet = Conv3D(self.DiscriminatorKernel*1, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer=lecun_normal(), 
                        data_format='channels_first', 
                        name=name+'_conv_dis_1')(inputs_embed)
        disnet = LeakyReLU(0.01)(disnet)
        
        # Hidden 1 : 64
        disnet = Conv3D(self.DiscriminatorKernel*2, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer=lecun_normal(), 
                        data_format='channels_first', 
                        name=name+'_conv_dis_2')(disnet)
        disnet = LeakyReLU(0.01)(disnet) 
        
        # Hidden 2 : 32
        disnet = Conv3D(self.DiscriminatorKernel*4, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer=lecun_normal(), 
                        data_format='channels_first', 
                        name=name+'_conv_dis_3')(disnet)
        disnet = LeakyReLU(0.01)(disnet)
        
        # Hidden 3 : 16
        disnet = Conv3D(self.DiscriminatorKernel*8, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer=lecun_normal(),
                        data_format='channels_first', 
                        name=name+'_conv_dis_4')(disnet)
        disnet = LeakyReLU(0.01)(disnet)
        
        # Hidden 4 : 8
        disnet = Conv3D(self.DiscriminatorKernel*16, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer=lecun_normal(),
                        data_format='channels_first', 
                        name=name+'_conv_dis_5')(disnet)
        disnet = LeakyReLU(0.01)(disnet)
        
        # Hidden 5 : 4
        disnet = Conv3D(self.DiscriminatorKernel*32, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer=lecun_normal(),
                        data_format='channels_first', 
                        name=name+'_conv_dis_6')(disnet)
        disnet = LeakyReLU(0.01)(disnet)
             
        # Decision : 2
        decision = Conv3D(1, 2, strides=1, 
                          use_bias=False,
                          kernel_initializer=lecun_normal(),
                          data_format='channels_first', 
                          name='dis_decision')(disnet) 
        decision = Reshape((1,))(decision)
        
        model = Model(inputs=[inputs,EmbeddedClass],outputs=[decision],name=name)
        return model
    
    def generator_block(self,name):
        #
        inputs = Input(shape=(1, self.ImageRow, self.ImageColumn, self.ImageDepth))
        targeSparseClass = Input(shape=(1,), dtype='int32')
                
        EmbeddedClass = Flatten()(Embedding(2,1, embeddings_initializer=gen_initializer,
                                           name=name+'_embedding',)(targeSparseClass))     
        
        # Representation
        gennet = ReflectPadding3D(padding=3)(inputs)
        gennet = Conv3D(self.GeneratorKernel, 7, strides=1, kernel_initializer=gen_initializer, 
                        use_bias=False,
                        name=name+'_gen_conv1', 
                        data_format='channels_first')(gennet)
        gennet = ConditionalInstanceNormalization3D(name=name+'_gen_isnorm_conv1')([gennet, EmbeddedClass])
        gennet = Activation('relu')(gennet)

        # Downsampling 1
        gennet = ReflectPadding3D(padding=1)(gennet)
        gennet = Conv3D(self.GeneratorKernel*2, 3, strides=2, kernel_initializer=gen_initializer, 
                        use_bias=False,
                        name=name+'_gen_conv2', 
                        data_format='channels_first')(gennet)
        gennet = ConditionalInstanceNormalization3D(name=name+'_gen_isnorm_conv2')([gennet, EmbeddedClass])
        gennet = Activation('relu')(gennet)
        
        # Downsampling 2
        gennet = ReflectPadding3D(padding=1)(gennet)
        gennet = Conv3D(self.GeneratorKernel*4, 3, strides=2, kernel_initializer=gen_initializer, 
                        use_bias=False,
                        name=name+'_gen_conv3', 
                        data_format='channels_first')(gennet)
        gennet = ConditionalInstanceNormalization3D(name=name+'_gen_isnorm_conv3')([gennet, EmbeddedClass])
        gennet = Activation('relu')(gennet)
               
        # Resnet blocks : 6, 8*4 = 32
        gennet = resnet_blocks(gennet, EmbeddedClass, self.GeneratorKernel*4, name=name+'_gen_block1')
        gennet = resnet_blocks(gennet, EmbeddedClass, self.GeneratorKernel*4, name=name+'_gen_block2')
        gennet = resnet_blocks(gennet, EmbeddedClass, self.GeneratorKernel*4, name=name+'_gen_block3')
        gennet = resnet_blocks(gennet, EmbeddedClass, self.GeneratorKernel*4, name=name+'_gen_block4')
        gennet = resnet_blocks(gennet, EmbeddedClass, self.GeneratorKernel*4, name=name+'_gen_block5')
        gennet = resnet_blocks(gennet, EmbeddedClass, self.GeneratorKernel*4, name=name+'_gen_block6')
        
        # Upsampling 1
        gennet = UpSampling3D(size=(2, 2, 2), 
                              data_format='channels_first')(gennet)
        gennet = ReflectPadding3D(padding=1)(gennet)
        gennet = Conv3D(self.GeneratorKernel*2, 3, strides=1, kernel_initializer=gen_initializer, 
                        use_bias=False,
                        name=name+'_gen_deconv1', 
                        data_format='channels_first')(gennet)
        gennet = ConditionalInstanceNormalization3D(name=name+'_gen_isnorm_deconv1')([gennet, EmbeddedClass])
        gennet = Activation('relu')(gennet)
        
        # Upsampling 2
        gennet = UpSampling3D(size=(2, 2, 2), 
                              data_format='channels_first')(gennet)
        gennet = ReflectPadding3D(padding=1)(gennet)
        gennet = Conv3D(self.GeneratorKernel, 3, strides=1, kernel_initializer=gen_initializer,
                        use_bias=False,
                        name=name+'_gen_deconv2', 
                        data_format='channels_first')(gennet)
        gennet = ConditionalInstanceNormalization3D(name=name+'_gen_isnorm_deconv2')([gennet, EmbeddedClass])
        gennet = Activation('relu')(gennet)
        
        # Reconstruction
        gennet = ReflectPadding3D(padding=3)(gennet)
        gennet = Conv3D(1, 7, strides=1, kernel_initializer=gen_initializer, 
                        use_bias=False,
                        name=name+'_gen_1conv', 
                        data_format='channels_first')(gennet)
        
        predictions = gennet
        predictions = Activation('tanh')(predictions)
        
        model = Model(inputs=[inputs,targeSparseClass],outputs=[predictions,EmbeddedClass],name=name)
        return model
    
    def generator(self):
        if self.G:
            return self.G
        
        self.G = self.generator_block('G')
        return self.G
                       
    def discriminator(self):
        if self.D:
            return self.D
        
        self.D = self.discriminator_block('DX')
        return self.D
    
    def generator_model(self):
        if self.GenModel:
            return self.GenModel
        
        print "We freeze the weights of Discriminator by setting their learning rate as 0 when updating Generator !"
        # We freeze the weights of Discriminator by setting their learning rate as 0 when updating Generator !
        AllParameters = 66
        GeneratorParameters = 53
        multipliers = np.ones(AllParameters)
        for idx in range(GeneratorParameters,AllParameters):
            multipliers[idx]= 0.0
            
        # Input
        input_gen = Input(shape=(1, self.ImageRow, self.ImageColumn, self.ImageDepth), name='input_gen')    
        domainSparseClass = Input(shape=(1,), name='domain_sparse_gen')
        targetSparseClass = Input(shape=(1,), name='target_sparse_gen')  
        
        # Network
        Gx_gen, EmbeddedClass = self.generator()([input_gen,targetSparseClass])     # Fake X      
        fool_decision = self.discriminator()([Gx_gen,EmbeddedClass])                # Fooling D
        Gx_gen_gen, _ = self.generator()([Gx_gen,domainSparseClass])                # Cycle X 
        
        # Model
        self.GenModel = Model([input_gen,targetSparseClass,domainSparseClass],
                              [fool_decision,Gx_gen_gen,Gx_gen])
        self.GenModel.compile(LR_Adam(lr=self.lr_GenModel, beta_1=0.5, beta_2=0.999, multipliers=multipliers),
                            loss=[wasserstein_loss, charbonnier_loss, total_variation_loss_3D],
                            loss_weights=[self.lamb_adv, self.lamb_rec, self.lamb_tv])
         
        return self.GenModel
    
    def discriminator_model(self):
        if self.DisModel:
            return self.DisModel 
        
        # Input
        real_dis = Input(shape=(1, self.ImageRow, self.ImageColumn, self.ImageDepth), name='real_dis')
        fake_dis = Input(shape=(1, self.ImageRow, self.ImageColumn, self.ImageDepth), name='fake_dis')       
        interp_dis = Input(shape=(1, self.ImageRow, self.ImageColumn, self.ImageDepth), name='interp_dis') 
        original_sparse_dis = Input(shape=(1,), name='original_sparse_dis')
        
        # Discriminator
        real_decision = self.discriminator()([real_dis,original_sparse_dis])          # Real X   
        fake_decision = self.discriminator()([fake_dis,original_sparse_dis])          # Fake X
        interp_decision = self.discriminator()([interp_dis,original_sparse_dis])  
                
        # GP loss
        partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=interp_dis,
                          gradient_penalty_weight=self.lamb_gp)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error*
        
        # Model
        self.DisModel = Model([real_dis,fake_dis,interp_dis,original_sparse_dis], 
                              [real_decision, fake_decision,interp_decision])
        self.DisModel.compile(Adam(lr=self.lr_DisModel, beta_1=0.5, beta_2=0.999),
                            loss=[wasserstein_loss,wasserstein_loss,partial_gp_loss],
                            loss_weights=[1,1,self.lamb_gp])            
        
        return self.DisModel
