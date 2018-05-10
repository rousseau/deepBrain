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
from sys import stdout
from ast import literal_eval as make_tuple

# Turn off Caffe's warning !
import os
os.environ["GLOG_minloglevel"] = "2"
import caffe   

def SRCNN3D_CPU(DeployNet,caffemodel,InterpolatedImage,layers=3):
    '''
        Caffe support data [N0ofFilter,Channel,Height,Width,Depth]
        -----------------
        DeployNet : text file,
            file to retrive training parameters of Caffe
        caffemodel : text file,
            file where Caffe stored parameters of each iteration
        layers : integer (default = 3)
            number of layers of CNN
        InterpolatedImage : 3d array
            interpolated low-resolution image (ILR image)
        
    '''
    # load directly CNN model parameters
    net = caffe.Net(DeployNet,caffemodel,caffe.TEST)
    
    # Size of interpolated low-resolution image (ILR image)
    [height, width, depth] = InterpolatedImage.shape
    
    # Intialization 
    InputDataFilter = np.zeros((1, height, width, depth))
    InputDataFilter[0,:,:,:] = InterpolatedImage
    
    # Feed ILR image to each layer
    for idx in range(0,layers):
        idx = idx+1
        # retrive parameter of idx th layer
        # weight
        WeightFilter = net.params['conv'+str(idx)][0].data    
        # bias
        BiasFilter = net.params['conv'+str(idx)][1].data    
        # fnum : Number of filters, channel: number of channels, FilterSize : filter's size           
        NumberFilter , NumberChannel, FilterSize, FilterSize, FilterSize = WeightFilter.shape
        # Print layer
        stdout.write("\rLayer : %d" % (idx))
        stdout.flush()
        
        # Result of idx th layer
        OutputWeightFilter = np.zeros((NumberFilter, height, width, depth))                 # Intialization 
        for i in range(0,NumberFilter):
            for j in range(0,NumberChannel):
                TempWeightFilter =  WeightFilter[i,j,:,:,:]
                OutputWeightFilter[i,:,:,:] = OutputWeightFilter[i,:,:,:] + scipy.ndimage.filters.convolve(InputDataFilter[j,:,:,:], TempWeightFilter, mode='nearest')
            OutputWeightFilter[i,:,:,:] = OutputWeightFilter[i,:,:,:] + BiasFilter[i]
        if idx != layers:
            # Last layer doesnot contain ReLU and doesnot need to save data for loop
            OutputFilter = np.maximum(OutputWeightFilter ,0)
            InputDataFilter = np.zeros((NumberFilter,height, width, depth))
            InputDataFilter = OutputFilter
    
    stdout.write("\n") 
    EstimatedImage = OutputWeightFilter[0,:,:,:]
    return EstimatedImage 

    
def SRCNN3D_GPU(DeployNet,caffemodel,InterpolatedImage):
    '''
        Caffe support data [N0ofFilter,Channel,Height,Width,Depth]
    ''' 
    import lasagne
    import lasagne.layers.dnn
    import theano
    import theano.tensor.nnet.conv3d2d

    # load CNN model parameters
    print ('Loading Caffemodel ....  ')
    net = caffe.Net(DeployNet,caffemodel,caffe.TEST)
    FilterWeight1 =  net.params['conv1'][0].data
    FilterBias1 =  net.params['conv1'][1].data
    FilterWeight2 =  net.params['conv2'][0].data
    FilterBias2 =  net.params['conv2'][1].data
    FilterWeight3 =  net.params['conv3'][0].data
    FilterBias3 =  net.params['conv3'][1].data
       
    # Padding the egdes of Interpolated Image before using SRCNN3D
    padding = (FilterWeight1.shape[-1]+FilterWeight2.shape[-1]+FilterWeight3.shape[-1]-3)/2    
    PadInterpolatedImage = np.lib.pad(InterpolatedImage, padding, 'edge')                 
    PadInterpolatedImage = PadInterpolatedImage[np.newaxis,np.newaxis,:,:,:]   # (in batch, in channel, row, column, depth )
    
    # Transfer to Theano Variables
    networkInputs = theano.shared(PadInterpolatedImage, borrow=True)           

    # SRCNN3D Network
    network = lasagne.layers.InputLayer(shape=(None,1,None,None,None),
                                        input_var=networkInputs)
    
    network = lasagne.layers.dnn.Conv3DDNNLayer(incoming = network ,               
                                          num_filters = FilterWeight1.shape[0], 
                                          filter_size = FilterWeight1.shape[-1], 
                                          stride=(1, 1, 1), 
                                          pad='valid', 
                                          untie_biases=False, 
                                          W=FilterWeight1,
                                          b=FilterBias1, 
                                          nonlinearity=lasagne.nonlinearities.rectify, 
                                          flip_filters=False)
    network = lasagne.layers.dnn.Conv3DDNNLayer(incoming = network ,               
                                          num_filters = FilterWeight2.shape[0], 
                                          filter_size = FilterWeight2.shape[-1], 
                                          stride=(1, 1, 1), 
                                          pad='valid', 
                                          untie_biases=False, 
                                          W=FilterWeight2,                          
                                          b=FilterBias2, 
                                          nonlinearity=lasagne.nonlinearities.rectify, 
                                          flip_filters=False)
    network = lasagne.layers.dnn.Conv3DDNNLayer(incoming = network ,               
                                          num_filters = FilterWeight3.shape[0], 
                                          filter_size = FilterWeight3.shape[-1], 
                                          stride=(1, 1, 1), 
                                          pad='valid', 
                                          untie_biases=False, 
                                          W=FilterWeight3,                           
                                          b=FilterBias3, 
                                          nonlinearity=None, 
                                          flip_filters=False)
    network = lasagne.layers.get_output(network)

    
    print ('Compiling and Executing ...')
    CompileFunction = theano.function([], network)
    EstimatedImage = CompileFunction()

    return EstimatedImage[0,0,:,:,:]    
    
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--test', help='Testing low-resolution image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-r', '--result', help='Estimated high-resolution image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-m', '--caffemodel', help='Name of caffe model (required)', type=str, required = True)
    parser.add_argument('-s', '--scale', help='Scale factor (default = 2,2,2)', type=str, default='2,2,2')
    parser.add_argument('-n', '--netdeploy', help='Name of train/test net protocol (default = SRCNN3D_deploy.prototxt)', type=str, default='SRCNN3D_deploy.prototxt')
    parser.add_argument('--order', help='Order of spline interpolation (default=3) ', type=int, default=3)
    parser.add_argument('-g', '--gpu', help='Using GPU : True or False (default=True) ', type=str, default='True')
     
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
        
        # spline interpolation 
        InterpolatedImage = scipy.ndimage.zoom(TestImageNorm, 
                                  zoom = Scale,
                                  order = args.order)  
        # SRCNN3D
        DeployNet = args.netdeploy
        caffemodel = args.caffemodel
        if args.gpu == 'True':
            EstimatedHRImage = SRCNN3D_GPU(DeployNet,caffemodel,InterpolatedImage)
        elif args.gpu == 'False':
            EstimatedHRImage = SRCNN3D_CPU(DeployNet,caffemodel,InterpolatedImage)
        else:
            raise AssertionError, 'Using GPU : True or False (default=True) !'
        # Save result
        OutFile = args.result[i]
        print 'SR image resust  : ', OutFile 
        
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

