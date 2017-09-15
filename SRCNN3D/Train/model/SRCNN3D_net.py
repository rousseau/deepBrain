# -*- coding: utf-8 -*-
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

import caffe
from caffe import layers as L

def SRCNN3D(hdf5name, batch_size, kernel ):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, 
                                 source=hdf5name, 
                                 ntop=2, 
                                 include=dict(phase = caffe.TRAIN))
    n.conv1 = L.Convolution(n.data, kernel_size=kernel[0], num_output=64, stride = 1, pad = 0,
                            param=[{'lr_mult':1},{'lr_mult':0.1}],
                            weight_filler=dict(type='gaussian',std=0.001),
                            bias_filler = dict(type= "constant", value=0),
                            engine = 1 ) 
    n.relu1 = L.ReLU(n.conv1, 
                     in_place=True,
                     engine = 1)
    n.conv2 = L.Convolution(n.conv1, kernel_size=kernel[1], num_output=32, stride = 1, pad = 0,
                            param=[{'lr_mult':1},{'lr_mult':0.1}],
                            weight_filler=dict(type='gaussian',std=0.001),
                            bias_filler = dict(type= "constant", value=0),
                            engine = 1 )                 
    n.relu2 = L.ReLU(n.conv2, 
                     in_place=True,
                     engine = 1)
    n.conv3 = L.Convolution(n.conv2, kernel_size=kernel[2], num_output=1, stride = 1, pad = 0,
                            param=[{'lr_mult':1},{'lr_mult':0.1}],
                            weight_filler=dict(type='gaussian',std=0.001),
                            bias_filler = dict(type= "constant", value=0),   
                            engine = 1 )    
    n.conv3_flat = L.Flatten(n.conv3)        
    n.label_flat = L.Flatten(n.label)
    n.loss = L.EuclideanLoss(n.conv3_flat,n.label_flat)                 
    return n.to_proto()

def SRCNN3D_deploy(netname, deployname):
    # =========== Writing SRReCNN3D_net deploy =================
    # Read content of Net and remove a part
    findStringBegin ="""
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
"""    
    findStringEnd ="""
layer {
  name: "label_flat"
  type: "Flatten"
  bottom: "label"
  top: "label_flat"
}
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "conv3_flat"
  bottom: "label_flat"
  top: "loss"
}
"""
    with open(netname) as f:
        contentNet = f.read()
        contentNet = contentNet[contentNet.find(findStringBegin):contentNet.find(findStringEnd)] # Removing unnecessary string
    with open(deployname, "w") as f1:
        f1.write(""" # This file is to deploy the parameters of SRCNN3D_net without reading HDF5 files
name: 'SRReCNN3D_net'
input: "data"
input_shape {
  dim: 1
  dim: 1
  dim: 21
  dim: 21
  dim: 21
}

""")
        f1.write(contentNet)
    return 1
    
