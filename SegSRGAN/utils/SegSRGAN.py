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

from keras.models import Model
from keras.layers import Input, LeakyReLU, Reshape
from keras.layers import Conv3D, Add, UpSampling3D, Activation
from keras.optimizers import Adam
from keras.initializers import lecun_normal
gen_initializer = lecun_normal() 

import sys
sys.path.insert(0, './utils')
from layers import wasserstein_loss, ReflectPadding3D, gradient_penalty_loss, InstanceNormalization3D, activation_SegSRGAN, charbonnier_loss
from Adam_lr_mult import LR_Adam

def resnet_blocks(input_res, kernel, name):

    in_res_1 = ReflectPadding3D(padding=1)(input_res)
    out_res_1 = Conv3D(kernel, 3, strides=1, kernel_initializer=gen_initializer, 
                       use_bias=False,
                       name=name+'_conv_a', 
                       data_format='channels_first')(in_res_1)
    out_res_1 = InstanceNormalization3D(name=name+'_isnorm_a')(out_res_1)
    out_res_1 = Activation('relu')(out_res_1)
    
    in_res_2 = ReflectPadding3D(padding=1)(out_res_1)
    out_res_2 = Conv3D(kernel, 3, strides=1, kernel_initializer=gen_initializer, 
                       use_bias=False,
                       name=name+'_conv_b', 
                       data_format='channels_first')(in_res_2)
    out_res_2 = InstanceNormalization3D(name=name+'_isnorm_b')(out_res_2)
    
    out_res = Add()([out_res_2,input_res])
    return out_res

class SegSRGAN(object):
    def __init__(self, ImageRow=64, ImageColumn=64, ImageDepth=64, 
                 FirstDiscriminatorKernel = 32, FirstGeneratorKernel = 16,
                 lamb_rec = 1, lamb_adv = 0.001, lamb_gp = 10, 
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
        self.lr_DisModel = lr_DisModel
        self.lr_GenModel = lr_GenModel
        
    def discriminator_block(self, name):
        """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
        the input is real or generated. Unlike normal GANs, the output is not sigmoid and does not represent a probability!
        Instead, the output should be as large and negative as possible for generated inputs and as large and positive
        as possible for real inputs.
        Note that the improved WGAN paper suggests that BatchNormalization should not be used in the discriminator."""
    
        # In:
        inputs = Input(shape=(2, self.ImageRow, self.ImageColumn, self.ImageDepth), name='dis_input')
        
        # Input 64
        disnet = Conv3D(self.DiscriminatorKernel*1, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer='he_normal', 
                        data_format='channels_first', 
                        name=name+'_conv_dis_1')(inputs)
        disnet = LeakyReLU(0.01)(disnet)
        
        # Hidden 1 : 32
        disnet = Conv3D(self.DiscriminatorKernel*2, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer='he_normal', 
                        data_format='channels_first', 
                        name=name+'_conv_dis_2')(disnet)
        disnet = LeakyReLU(0.01)(disnet) 
        
        # Hidden 2 : 16
        disnet = Conv3D(self.DiscriminatorKernel*4, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer='he_normal', 
                        data_format='channels_first', 
                        name=name+'_conv_dis_3')(disnet)
        disnet = LeakyReLU(0.01)(disnet)
        
        # Hidden 3 : 8
        disnet = Conv3D(self.DiscriminatorKernel*8, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer='he_normal',
                        data_format='channels_first', 
                        name=name+'_conv_dis_4')(disnet)
        disnet = LeakyReLU(0.01)(disnet)
        
        # Hidden 4 : 4
        disnet = Conv3D(self.DiscriminatorKernel*16, 4, strides=2, 
                        padding = 'same',
                        kernel_initializer='he_normal',
                        data_format='channels_first', 
                        name=name+'_conv_dis_5')(disnet)
        disnet = LeakyReLU(0.01)(disnet)
             
        # Decision : 2
        decision = Conv3D(1, 2, strides=1, 
                          use_bias=False,
                          kernel_initializer='he_normal',
                          data_format='channels_first', 
                          name='dis_decision')(disnet) 
        decision = Reshape((1,))(decision)
        
        model = Model(inputs=[inputs],outputs=[decision],name=name)
        return model
    
    def generator_block(self,name):
        #
        inputs = Input(shape=(1, self.ImageRow, self.ImageColumn, self.ImageDepth))

        # Representation
        gennet = ReflectPadding3D(padding=3)(inputs)
        gennet = Conv3D(self.GeneratorKernel, 7, strides=1, kernel_initializer=gen_initializer, 
                        use_bias=False,
                        name=name+'_gen_conv1', 
                        data_format='channels_first')(gennet)
        gennet = InstanceNormalization3D(name=name+'_gen_isnorm_conv1')(gennet)
        gennet = Activation('relu')(gennet)

        # Downsampling 1
        gennet = ReflectPadding3D(padding=1)(gennet)
        gennet = Conv3D(self.GeneratorKernel*2, 3, strides=2, kernel_initializer=gen_initializer, 
                        use_bias=False,
                        name=name+'_gen_conv2', 
                        data_format='channels_first')(gennet)
        gennet = InstanceNormalization3D(name=name+'_gen_isnorm_conv2')(gennet)
        gennet = Activation('relu')(gennet)
        
        # Downsampling 2
        gennet = ReflectPadding3D(padding=1)(gennet)
        gennet = Conv3D(self.GeneratorKernel*4, 3, strides=2, kernel_initializer=gen_initializer, 
                        use_bias=False,
                        name=name+'_gen_conv3', 
                        data_format='channels_first')(gennet)
        gennet = InstanceNormalization3D(name=name+'_gen_isnorm_conv3')(gennet)
        gennet = Activation('relu')(gennet)
               
        # Resnet blocks : 6, 8*4 = 32
        gennet = resnet_blocks(gennet, self.GeneratorKernel*4, name=name+'_gen_block1')
        gennet = resnet_blocks(gennet, self.GeneratorKernel*4, name=name+'_gen_block2')
        gennet = resnet_blocks(gennet, self.GeneratorKernel*4, name=name+'_gen_block3')
        gennet = resnet_blocks(gennet, self.GeneratorKernel*4, name=name+'_gen_block4')
        gennet = resnet_blocks(gennet, self.GeneratorKernel*4, name=name+'_gen_block5')
        gennet = resnet_blocks(gennet, self.GeneratorKernel*4, name=name+'_gen_block6')
        
        # Upsampling 1
        gennet = UpSampling3D(size=(2, 2, 2), 
                              data_format='channels_first')(gennet)
        gennet = ReflectPadding3D(padding=1)(gennet)
        gennet = Conv3D(self.GeneratorKernel*2, 3, strides=1, kernel_initializer=gen_initializer, 
                        use_bias=False,
                        name=name+'_gen_deconv1', 
                        data_format='channels_first')(gennet)
        gennet = InstanceNormalization3D(name=name+'_gen_isnorm_deconv1')(gennet)
        gennet = Activation('relu')(gennet)
        
        # Upsampling 2
        gennet = UpSampling3D(size=(2, 2, 2), 
                              data_format='channels_first')(gennet)
        gennet = ReflectPadding3D(padding=1)(gennet)
        gennet = Conv3D(self.GeneratorKernel, 3, strides=1, kernel_initializer=gen_initializer,
                        use_bias=False,
                        name=name+'_gen_deconv2', 
                        data_format='channels_first')(gennet)
        gennet = InstanceNormalization3D(name=name+'_gen_isnorm_deconv2')(gennet)
        gennet = Activation('relu')(gennet)
        
        # Reconstruction
        gennet = ReflectPadding3D(padding=3)(gennet)
        gennet = Conv3D(2, 7, strides=1, kernel_initializer=gen_initializer, 
                        use_bias=False,
                        name=name+'_gen_1conv', 
                        data_format='channels_first')(gennet)
        
        predictions = gennet
        predictions = activation_SegSRGAN()([predictions,inputs])
        
        model = Model(inputs=inputs,outputs=predictions,name=name)
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
        AllParameters = 63
        GeneratorParameters = 52
        multipliers = np.ones(AllParameters)
        for idx in range(GeneratorParameters,AllParameters):
            multipliers[idx]= 0.0
            
        # Input
        input_gen = Input(shape=(1, self.ImageRow, self.ImageColumn, self.ImageDepth), name='input_gen')    

        # Network
        Gx_gen = self.generator()(input_gen)               # Fake X      
        fool_decision = self.discriminator()(Gx_gen)      # Fooling D
        
        # Model
        self.GenModel = Model(input_gen,[fool_decision,Gx_gen])
        self.GenModel.compile(LR_Adam(lr=self.lr_GenModel, beta_1=0.5, beta_2=0.999, multipliers=multipliers),
                            loss=[wasserstein_loss, charbonnier_loss],
                            loss_weights=[self.lamb_adv, self.lamb_rec])
         
        return self.GenModel
    
    def discriminator_model(self):
        if self.DisModel:
            return self.DisModel 
        
        # Input
        real_dis = Input(shape=(2, self.ImageRow, self.ImageColumn, self.ImageDepth), name='real_dis')
        fake_dis = Input(shape=(2, self.ImageRow, self.ImageColumn, self.ImageDepth), name='fake_dis')       
        interp_dis = Input(shape=(2, self.ImageRow, self.ImageColumn, self.ImageDepth), name='interp_dis') 
                
        # Discriminator
        real_decision = self.discriminator()(real_dis)            # Real X   
        fake_decision = self.discriminator()(fake_dis)            # Fake X
        interp_decision = self.discriminator()(interp_dis)        # interpolation X
        
        # GP loss
        partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=interp_dis,
                          gradient_penalty_weight=self.lamb_gp)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error*
        
        # Model
        self.DisModel = Model([real_dis,fake_dis,interp_dis], [real_decision, fake_decision,interp_decision])
        self.DisModel.compile(Adam(lr=self.lr_DisModel, beta_1=0.5, beta_2=0.999),
                            loss=[wasserstein_loss,wasserstein_loss,partial_gp_loss],
                            loss_weights=[1,1,self.lamb_gp])            
        
        return self.DisModel
