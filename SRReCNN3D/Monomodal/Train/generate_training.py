
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
import sys
from ast import literal_eval as make_tuple

sys.path.insert(0, './utils')
sys.path.insert(0, './model')
from utils3d import shave3D, imadjust3D, modcrop3D
from store2hdf5 import store2hdf53D
from patches import array_to_patches
from SRReCNN3D_net import SRReCNN3D_net, SRReCNN3D_deploy

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--reference', help='Reference image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-o', '--output', help='Name of output HDF5 files (required)', type=str, action='append', required = True)
    parser.add_argument('-s', '--scale',  help='Scale factor (default = 2,2,2). Append mode: -s 2,2,2 -s 3,3,3 ', type=str, action='append')
    parser.add_argument('--stride', help='Indicates step size at which extraction shall be performed (default=10)', type=int, default=10)
    parser.add_argument('-p','--patchsize', help='Indicates input patch size for extraction', type=int, default=21)
    parser.add_argument('-b','--batch', help='Indicates batch size for HDF5 storage', type=int, default=64)
    parser.add_argument('-l','--layers', help='Indicates number of layers of network (default=10)', type=int, default=10)
    parser.add_argument('-k','--kernel', help='Indicates size of filter (default=3)', type=int, default=3)
    parser.add_argument('--numkernel', help='Indicates number of filters (default=64)', type=int, default=64)
    parser.add_argument('-r','--residual', help='Using residual learning or None (default=True)', type=str, default='True')
    parser.add_argument('--border', help='Border to remove (default=10,10,0)', type=str, default='10,10,0')
    parser.add_argument('--order', help='Order of spline interpolation (default=3) ', type=int, default=3)
    parser.add_argument('--samples', help='Indicates limit of samples in HDF5 file (optional)', type=int)
    parser.add_argument('--sigma', help='Standard deviation (sigma) of Gaussian blur (default=1)', type=int, default=1)
    parser.add_argument('-t', '--text', help='Name of a text (.txt) file which contains HDF5 file names (default: model/train.txt)', type=str, default='model/train.txt')
    parser.add_argument('-n', '--netname', help='Name of train netwotk protocol (default=model/SRReCNN3D_net.prototxt)', type=str, default='model/SRReCNN3D_net.prototxt')
    parser.add_argument('-d', '--deployname', help='Name of deploy files in order to deploy the parameters of SRReCNN3D_net without reading HDF5 files (default=model/SRReCNN3D_deploy.prototxt)', type=str, default='model/SRReCNN3D_deploy.prototxt')
    
    args = parser.parse_args()
    
    #  ==== Parser  ===
    # Check number of input and output name:
    if len(args.reference) != len(args.output) :   
        raise AssertionError, 'Number of inputs and outputs should be matched !'    
        
    PatchSize = args.patchsize
    padding = int((args.kernel - 1)/float(2))
    
    # Check scale
    if args.scale is None:
        args.scale = [(2,2,2)]
    else:
        for idx in range(0,len(args.scale)):
            args.scale[idx] = make_tuple(args.scale[idx])
            if np.isscalar(args.scale[idx]):
                args.scale[idx] = (args.scale[idx],args.scale[idx],args.scale[idx])
            else:
                if len(args.scale[idx])!=3:
                    raise AssertionError, 'Not support this scale factor !'  
    
    # Check residual learning mode
    if args.residual == 'True':
        residual = True
    elif args.residual == 'False':
        residual = False
    else:
        raise AssertionError, 'Not support this residual mode. Try True or False !' 
                    
    # Check border removing
    border = make_tuple(args.border)
    if np.isscalar(border):
        border = (border,border,border)
    else:
        if len(border)!=3:
            raise AssertionError, 'Not support this scale factor !'       
        
    # Writing a text (.txt) file which contains HDF5 file names 
    OutFile = open(str(args.text), "w")
    
    # ============ Processing images ===========================================

    for i in range(0,len(args.reference)):
        # initialization : n-dimensional Caffe supports data's form : [numberOfBatches,channels,heigh,width,depth]  
        HDF5Datas = []
        HDF5Labels = []
        
        # Read reference image
        ReferenceName = args.reference[i]
        print '================================================================'
        print 'Processing image : ', ReferenceName
        # Read NIFTI
        ReferenceNifti = sitk.ReadImage(ReferenceName)
        
        # Get data from NIFTI
        ReferenceImage = np.swapaxes(sitk.GetArrayFromImage(ReferenceNifti),0,2).astype('float32')
        
        # Normalization 
        ReferenceImage =  imadjust3D(ReferenceImage,[0,1])    
        
        # ===== Generate input LR image =====
        # Blurring
        BlurReferenceImage = scipy.ndimage.filters.gaussian_filter(ReferenceImage,
                                                            sigma = args.sigma)
                                                            
        for scale in args.scale:
            print 'With respect to scale factor x', scale, ' : '
            
            # Modcrop to scale factor
            BlurReferenceImage = modcrop3D(BlurReferenceImage,scale)
            ReferenceImage = modcrop3D(ReferenceImage,scale)
            
            # Downsampling
            LowResolutionImage = scipy.ndimage.zoom(BlurReferenceImage,
                                      zoom = (1/float(idxScale) for idxScale in scale),
                                      order = args.order)  
            
            # Cubic Interpolation     
            InterpolatedImage = scipy.ndimage.zoom(LowResolutionImage, 
                                      zoom = scale,
                                      order = args.order)  
                                  
            # Shave border
            LabelImage = shave3D(ReferenceImage, border)    
            DataImage = shave3D(InterpolatedImage, border)   
            
            # Extract 3D patches
            DataPatch = array_to_patches(DataImage, 
                                         patch_shape=(PatchSize,PatchSize,PatchSize), 
                                         extraction_step = args.stride , 
                                         normalization=False)
            print 'for the interpolated low-resolution patches of training phase.'                                 
            LabelPatch = array_to_patches(LabelImage, 
                                         patch_shape=(PatchSize,PatchSize,PatchSize), 
                                         extraction_step = args.stride , 
                                         normalization=False)
            print 'for the reference high-resolution patches of training phase.'                                  
            # Append array
            HDF5Datas.append(DataPatch)
            HDF5Labels.append(LabelPatch)
        
        
        # List type to array numpy
        HDF5Datas = np.asarray(HDF5Datas).reshape(-1,PatchSize,PatchSize,PatchSize)
        HDF5Labels = np.asarray(HDF5Labels).reshape(-1,PatchSize,PatchSize,PatchSize)
        
         # Add channel axis !  
        HDF5Datas = HDF5Datas[:,np.newaxis,:,:,:]
        HDF5Labels = HDF5Labels[:,np.newaxis,:,:,:]
            
        # Rearrange
        np.random.seed(0)       # makes the random numbers predictable
        RandomOrder = np.random.permutation(HDF5Datas.shape[0])
        HDF5Datas = HDF5Datas[RandomOrder,:,:,:,:]
        HDF5Labels = HDF5Labels[RandomOrder,:,:,:,:]
        
        # ============================================================================================
        # Crop data to desired number of samples
        if args.samples :
            HDF5Datas = HDF5Datas[:args.samples ,:,:,:,:]
            HDF5Labels = HDF5Labels[:args.samples ,:,:,:,:]
            
        # Writing to HDF5   
        hdf5name = args.output[i]
        print '*) Writing to HDF5 file : ', hdf5name
        StartLocation = {'dat':(0,0,0,0,0), 'lab': (0,0,0,0,0)}
        CurrentDataLocation = store2hdf53D(filename=hdf5name, 
                                           datas=HDF5Datas, 
                                           labels=HDF5Labels, 
                                           startloc=StartLocation, 
                                           chunksz=args.batch )
                                   
        # Reading HDF5 file                           
        import h5py
        with h5py.File(hdf5name,'r') as hf:
            udata = hf.get('data')
            print 'Shape of interpolated low-resolution patches:', udata.shape
            print 'Chunk (batch) of interpolated low-resolution patches:', udata.chunks
            ulabel = hf.get('label')
            print 'Shape of reference high-resolution patches:', ulabel.shape
            print 'Chunk (batch) of reference high-resolution patches:', ulabel.chunks

        # Writing a text file which contains HDF5 file names 
        OutFile.write(hdf5name)
        OutFile.write('\n')
        
    # =========== Wrinting net ==================  
    with open(args.netname , 'w') as f:
        f.write(str(SRReCNN3D_net(args.text, args.batch, args.layers, args.kernel, args.numkernel, padding, residual)))
    SRReCNN3D_deploy(args.netname, args.deployname)
