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
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv3D, Add, Activation
from Adam_lr_mult import LR_Adam

class SRReCNN3D(object):
    def __init__(self, ImageRow=25, ImageColumn=25, ImageDepth=25, 
                 channel=1, NetDepth=20, NetNumKernel=64,
                 KernelSize=3, LearningRate=0.0001, Residual=True):
        self.ImageRow = ImageRow
        self.ImageColumn = ImageColumn
        self.ImageDepth = ImageDepth
        self.channel = channel
        self.NetNumKernel = NetNumKernel
        self.NetDepth = NetDepth
        self.Residual = Residual
        self.KernelSize = KernelSize
        self.LearningRate = LearningRate
        self.Generator = None   # generator

    def generator(self):
        if self.Generator:
            return self.Generator

        # Multiplier for weights [0,2,4,6,...] (=1) and bias [1,3,5,7,...] (=0.1)
        # If lr = 0.001 , lr for weights is 0.001 and bias is 0.0001
        multipliers = np.zeros(self.NetDepth*2)
        for idx in range(self.NetDepth*2):
            if idx % 2:
                multipliers[idx]=0.1        # Bias
            else:
                multipliers[idx]=1          # Weight        
        
        input_shape = (self.channel, self.ImageRow, self.ImageColumn, self.ImageDepth)
        inputs = Input(shape=input_shape)
        
        gennet = Conv3D(self.NetNumKernel, self.KernelSize, strides=1, 
                        kernel_initializer= 'he_normal', 
                        padding='same', name='SR_gen_conv1', 
                        data_format='channels_first')(inputs)
        gennet = Activation('relu')(gennet)
        
        for layer in range(2,self.NetDepth):
            gennet = Conv3D(self.NetNumKernel, self.KernelSize, strides=1, 
                            kernel_initializer= 'he_normal', 
                            padding='same', name='SR_gen_conv'+str(layer), 
                            data_format='channels_first')(gennet)
            gennet = Activation('relu')(gennet)
               
        gennet = Conv3D(1, self.KernelSize, strides=1, 
                        kernel_initializer= 'he_normal', 
                        padding='same', name='SR_gen_conv'+str(self.NetDepth), 
                        data_format='channels_first')(gennet)
        
        if self.Residual:
            predictions = Add()([gennet,inputs])
        else:
            predictions = gennet

        self.Generator = Model(inputs=inputs,outputs=predictions)
        optimizer = LR_Adam(lr=self.LearningRate,multipliers=multipliers)
        self.Generator.compile(loss='mean_squared_error', optimizer=optimizer)
        return self.Generator
