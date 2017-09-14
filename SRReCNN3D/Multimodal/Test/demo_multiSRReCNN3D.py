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


import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from ast import literal_eval as make_tuple

# Turn off Caffe's warning !
import os
os.environ["GLOG_minloglevel"] = "2"
import caffe   
import lasagne
import lasagne.layers.dnn
import theano
import theano.tensor.nnet.conv3d2d
    
def MultiSRReCNN3D(DeployNet,caffemodel,InterpolatedImage,ReferenceImage,layers,residual=True):
    '''
        Caffe support data [N0ofFilter,Channel,Height,Width,Depth]
        DeployNet : text file,
            file to retrive training parameters of Caffe
        caffemodel : text file,
            file where Caffe stored parameters of each iteration
        layers : integer,
            number of layers of CNN
        InterpolatedImage : 3d array
            interpolated low-resolution image (ILR image)
        ReferenceImage : 3d array
            high resolution reference image (HR reference)
            
    ''' 
    # load CNN model parameters
    net = caffe.Net(DeployNet,caffemodel,caffe.TEST)                  
    InterpolatedImage5D = InterpolatedImage[np.newaxis,np.newaxis,:,:,:]                # (in batch, in channel, row, column, depth )
    ReferenceImage5D = ReferenceImage[np.newaxis,np.newaxis,:,:,:] 
    
    # Transfer to Theano Variables
    InterpolatedImageTensor = theano.shared(InterpolatedImage5D, borrow=True)           
    ReferenceImageTensor = theano.shared(ReferenceImage5D, borrow=True)
    
    # Network
    InterpolatedImageInput = lasagne.layers.InputLayer(shape=(None,1,None,None,None),
                                                       input_var=InterpolatedImageTensor)
    ReferenceImageInput = lasagne.layers.InputLayer(shape=(None,1,None,None,None),
                                                       input_var=ReferenceImageTensor)
    
    network = lasagne.layers.ConcatLayer(incomings=(InterpolatedImageInput,ReferenceImageInput), 
                                         axis=1, cropping=None)
    
    # 1st - (n-1)th layer
    for idx in range(1,layers):
        network = lasagne.layers.dnn.Conv3DDNNLayer(incoming = network ,               
                                          num_filters = net.params['conv'+str(idx)][0].data.shape[0], 
                                          filter_size = net.params['conv'+str(idx)][0].data.shape[-1], 
                                          stride=(1, 1, 1), 
                                          pad=1, 
                                          untie_biases=False, 
                                          W=net.params['conv'+str(idx)][0].data,
                                          b=net.params['conv'+str(idx)][1].data, 
                                          nonlinearity=lasagne.nonlinearities.rectify, 
                                          flip_filters=False)
    # n th layer
    network = lasagne.layers.dnn.Conv3DDNNLayer(incoming = network ,               
                                          num_filters = net.params['conv'+str(layers)][0].data.shape[0], 
                                          filter_size = net.params['conv'+str(layers)][0].data.shape[-1], 
                                          stride=(1, 1, 1), 
                                          pad=1, 
                                          untie_biases=False, 
                                          W=net.params['conv'+str(layers)][0].data,                           
                                          b=net.params['conv'+str(layers)][1].data, 
                                          nonlinearity=None, 
                                          flip_filters=False)
    
    network = lasagne.layers.get_output(network)
    
    # Compiling
    print ('Compiling and Executing ...')
    CompileFunction = theano.function([], network)
    EstimatedImage = CompileFunction()
    
    if residual == True:
        SRImage = EstimatedImage[0,0,:,:,:]+InterpolatedImage
    else:
        SRImage = EstimatedImage[0,0,:,:,:]
    return SRImage
    
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--test', help='Testing low-resolution image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-f', '--reference', help='High-resolution image reference filename (required)', type=str, action='append', required = True)
    parser.add_argument('-r', '--result', help='Estimated high-resolution image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-m', '--caffemodel', help='Name of caffe model (required)', type=str, required = True)
    parser.add_argument('-s', '--scale', help='Scale factor (default = 2,2,2)', type=str, default='2,2,2')
    parser.add_argument('-l', '--layers', help='Layer number of network (default = 10)', type=int, default=10)
    parser.add_argument('-n', '--netdeploy', help='Name of train/test net protocol (default = SRCNN3D_deploy.prototxt)', type=str, default='SRCNN3D_deploy.prototxt')
    parser.add_argument('--order', help='Order of spline interpolation (default=3) ', type=int, default=3)
    parser.add_argument('--residual', help='Using residual (Skip Connection) or None (default=True)', type=str, default='True')
     
    args = parser.parse_args()
    
    # Check scale factor type
    Scale = make_tuple(args.scale)
    if np.isscalar(Scale):
        Scale = (Scale,Scale,Scale)
    else:
        if len(Scale)!=3:
            raise AssertionError, 'Not support this scale factor !'  
            
    # Check number of test image and result name:
    if len(args.result) != len(args.test):   
        raise AssertionError, 'Number of test images and result names should be matched !'   
        
    for i in range(0,len(args.test)):
       
        # Read low-resolution image
        TestFile = args.test[i]
        print 'Processing testing image : ', TestFile 
        TestNifti = sitk.ReadImage(TestFile)
        TestImage = np.swapaxes(sitk.GetArrayFromImage(TestNifti),0,2).astype('float32')
        TestImageMinValue = float(np.min(TestImage))
        TestImageMaxValue = float(np.max(TestImage))
        TestImageNorm = TestImage/TestImageMaxValue 
        
        # Read HR reference
        ReferenceFile = args.reference[i]
        print 'Reference image : ', ReferenceFile 
        ReferenceNifti = sitk.ReadImage(ReferenceFile)
        ReferenceImage = np.swapaxes(sitk.GetArrayFromImage(ReferenceNifti),0,2).astype('float32')
        ReferenceImageMaxValue = float(np.max(ReferenceImage))
        ReferenceImageNorm = ReferenceImage/ReferenceImageMaxValue 
        
        # spline interpolation 
        InterpolatedImage = scipy.ndimage.zoom(TestImageNorm, 
                                                  zoom = Scale,
                                                  order = args.order)  
        # SRResidualCNN3D
        DeployNet = args.netdeploy
        caffemodel = args.caffemodel 
        if args.residual == 'True':
            EstimatedHRImage = MultiSRReCNN3D(DeployNet,caffemodel,InterpolatedImage,ReferenceImageNorm,args.layers,residual=True)
        elif args.residual == 'False':
            EstimatedHRImage = MultiSRReCNN3D(DeployNet,caffemodel,InterpolatedImage,ReferenceImageNorm,args.layers,residual=False)
        else:
            raise AssertionError, 'Using residual (Skip Connection) : True or False (default=True) !'
        
        # Affine change
        SliceDif = tuple(((idxScale-1)/2.0) for idxScale in Scale)
        NewSpacing  = tuple(itema/itemb for itema,itemb in zip(TestNifti.GetSpacing(),Scale))
        NewOrigin   = TestNifti.GetOrigin()
        NewDirection = TestNifti.GetDirection()
        
        EstimatedHRImageInverseNorm = EstimatedHRImage*TestImageMaxValue
        EstimatedHRImageInverseNorm[EstimatedHRImageInverseNorm <= TestImageMinValue] = TestImageMinValue    # Clear negative value
        OutputImage = sitk.GetImageFromArray(np.swapaxes(EstimatedHRImageInverseNorm,0,2))
        OutputImage.SetSpacing(NewSpacing)
        OutputImage.SetOrigin(NewOrigin)
        OutputImage.SetDirection(NewDirection)
        
        # Save result
        OutFile = args.result[i]
        print 'SR image resust  : ', OutFile 
        sitk.WriteImage(OutputImage,OutFile)
