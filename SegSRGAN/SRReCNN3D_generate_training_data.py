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
from utils3d import shave3D, modcrop3D
from store2hdf5 import store2hdf53D
from patches import array_to_patches

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--reference', help='Reference image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-o', '--output', help='Name of output HDF5 files (required)', type=str, action='append', required = True)
    parser.add_argument('-n', '--newlowres',  help='Desired low resolution (default = 0.4464,0.4464,3 mm)', type=str, default='0.4464,0.4464,3')
    parser.add_argument('--stride', help='Image extraction stride (default=10)', type=int, default=10)
    parser.add_argument('-p','--patchsize', help='Indicates input patch size for extraction', type=int, default=25)
    parser.add_argument('-b','--batch', help='Indicates batch size for HDF5 storage', type=int, default=64)
    parser.add_argument('--border', help='Border to remove (default=20,20,0)', type=str, default='20,20,0')
    parser.add_argument('--order', help='Order of spline interpolation (default=3) ', type=int, default=3)
    parser.add_argument('-s', '--samples', help='Number of samples in HDF5 file (optional)', type=int)
    parser.add_argument('-t', '--text', help='Name of a text (.txt) file which contains HDF5 file names (default: train.txt)', type=str, default='train.txt')
    
    args = parser.parse_args()
    
    #  ==== Parser  ===
    # Check number of input and output name:
    if len(args.reference) != len(args.output) :   
        raise AssertionError, 'Number of inputs and outputs should be matched !'    
    
    # Patch size
    PatchSize = args.patchsize
    
    # Check resolution
    NewResolution = make_tuple(args.newlowres)
    if np.isscalar(NewResolution):
        NewResolution = (NewResolution,NewResolution,NewResolution)
    else:
        if len(NewResolution)!=3:
            raise AssertionError, 'Not support this resolution !'  
    
    # Check sigma
    constant = 2*np.sqrt(2*np.log(2)) # As Greenspan et al. (Full_width_at_half_maximum : slice thickness)
    SigmaBlur = NewResolution/constant
    
    # Check border removing
    border = make_tuple(args.border)
    if np.isscalar(border):
        border = (border,border,border)
    else:
        if len(border)!=3:
            raise AssertionError, 'Not support this border !'       
            
    # Writing a text (.txt) file which contains HDF5 file names 
    OutFile = open(str(args.text), "w")
    
    # ============ Processing images ===========================================

    for i in range(0,len(args.reference)):
        
        # Read reference image
        ReferenceName = args.reference[i]
        print '================================================================'
        print 'Processing image : ', ReferenceName
        # Read NIFTI
        ReferenceNifti = sitk.ReadImage(ReferenceName)
        
        # Get data from NIFTI
        ReferenceImage = np.swapaxes(sitk.GetArrayFromImage(ReferenceNifti),0,2).astype('float32')
        
        # Get resolution to scaling factor
        UpScale = tuple(itemb/itema for itema,itemb in zip(ReferenceNifti.GetSpacing(),NewResolution))

        # Modcrop to scale factor
        ReferenceImage = modcrop3D(ReferenceImage,UpScale)

        # ===== Generate input LR image =====
        # Blurring
        BlurReferenceImage = scipy.ndimage.filters.gaussian_filter(ReferenceImage,
                                                            sigma = SigmaBlur)
                                                            
        print 'Generating LR images with the resolution of ', NewResolution
        
        # Downsampling
        LowResolutionImage = scipy.ndimage.zoom(BlurReferenceImage,
                                  zoom = (1/float(idxScale) for idxScale in UpScale),
                                  order = 0)  
        
        # Normalization by the max valeur of LR image
        MaxValue = np.max(LowResolutionImage)
        NormalizedReferenceImage =  ReferenceImage/MaxValue
        NormalizedLowResolutionImage =  LowResolutionImage/MaxValue
        
        # Cubic Interpolation     
        InterpolatedImage = scipy.ndimage.zoom(NormalizedLowResolutionImage, 
                                  zoom = UpScale,
                                  order = args.order)  
                              
        # Shave border
        LabelImage = shave3D(NormalizedReferenceImage, border)    
        DataImage = shave3D(InterpolatedImage, border)   
        
        # Extract 3D patches
        print 'Generating training patches with the resolution of ', NewResolution, ' : '
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
                             
        # n-dimensional Caffe supports data's form : [numberOfBatches,channels,heigh,width,depth]         
         # Add channel axis !  
        HDF5Datas = DataPatch[:,np.newaxis,:,:,:]
        HDF5Labels = LabelPatch[:,np.newaxis,:,:,:]
            
        # Rearrange
        np.random.seed(0)       # makes the random numbers predictable
        RandomOrder = np.random.permutation(HDF5Datas.shape[0])
        HDF5Datas = HDF5Datas[RandomOrder,:,:,:,:]
        HDF5Labels = HDF5Labels[RandomOrder,:,:,:,:]
        
        # ============================================================================================
        # Crop data to desired number of samples
        if args.samples :
            HDF5Datas = HDF5Datas[:args.samples ,...]
            HDF5Labels = HDF5Labels[:args.samples ,...]
            
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
            print 'Shape of input patches:', udata.shape
            ulabel = hf.get('label')
            print 'Shape of output patches:', ulabel.shape

        # Writing a text file which contains HDF5 file names 
        OutFile.write(hdf5name)
        OutFile.write('\n')
