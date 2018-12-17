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
import sys
import h5py
sys.path.insert(0, './utils')
from store2hdf5 import store2hdf53Dgan
from patches import array_to_patches

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-1', '--t1w', help='Reference T1-weighted image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-2', '--t2w', help='Reference T2-weighted image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-o1', '--out1', help='Name of output HDF5 files correspond T1-weighted images (required)', type=str, action='append', required = True)
    parser.add_argument('-o2', '--out2', help='Name of output HDF5 files correspond T2-weighted images (required)', type=str, action='append', required = True)
    parser.add_argument('--stride', help='Image extraction stride (default=10)', type=int, default=18)
    parser.add_argument('-b','--batch', help='Indicates batch size for HDF5 storage', type=int, default=1)
    parser.add_argument('-s', '--samples', help='Number of samples of a HDF5 file (optional)', type=int, default=20)
    parser.add_argument('--textT1', help='Text file contains HDF5 file names of T1-weighted images (default: trainT1.txt)', type=str, default='trainT1.txt')
    parser.add_argument('--textT2', help='Text file contains HDF5 file names of T2-weighted images (default: trainT2.txt)', type=str, default='trainT2.txt')
    parser.add_argument('--thresholdvalue', help='Value of dark region to remove (default = 0)', type=int, default=0)
    
    args = parser.parse_args()
    
    #  ==== Parser  ===
    # Check number of input and output name:
    if (len(args.t1w) != len(args.out1)) or (len(args.t1w) != len(args.t2w))  or (len(args.t2w) != len(args.out2))  :   
        raise AssertionError, 'Number of files should be matched !'    
    
    # Patch size is fixed as 128
    PatchSize = 128
            
    # =================================================================================================
    # Read t1w reference images
    # Writing a text (.txt) file which contains HDF5 file names 
    OutFile = open(str(args.textT1), "w")
    
    for i in range(0,len(args.t1w)):

        T1wImage = args.t1w[i]
        print '================================================================'
        print 'Processing T1w image : ', T1wImage
        # Read NIFTI
        T1wImageNifti = sitk.ReadImage(T1wImage)
        
        # Get data from NIFTI
        T1wImage = np.swapaxes(sitk.GetArrayFromImage(T1wImageNifti),0,2).astype('float32')
        
        # Normalization by the max valeur
        MaxValue = np.max(T1wImage)
        NormalizedT1wImage = T1wImage/MaxValue

        # Shave region outside
        print 'Remove the region outside the brain with the value of ', args.thresholdvalue
        darkRegionValue = args.thresholdvalue
        darkRegionBox = np.where(NormalizedT1wImage>darkRegionValue)   
        border = ((np.min(darkRegionBox[0]),np.max(darkRegionBox[0])),
                  (np.min(darkRegionBox[1]),np.max(darkRegionBox[1])),
                  (np.min(darkRegionBox[2]),np.max(darkRegionBox[2])))     
        DatasT1wImage = NormalizedT1wImage[border[0][0]:border[0][1],
                                           border[1][0]:border[1][1],
                                           border[2][0]:border[2][1]]    
        
        # Extract 3D patches                              
        DatasT1wPatch = array_to_patches(DatasT1wImage, 
                                         patch_shape=(PatchSize,PatchSize,PatchSize), 
                                         extraction_step = args.stride , 
                                         normalization=False)
        print 'for patches of training phase.'        
                          
        # n-dimensional Caffe supports data's form : [numberOfBatches,channels,heigh,width,depth]         
        # Add channel axis !  
        DatasT1wPatch = DatasT1wPatch[:,np.newaxis,:,:,:]
                
        # Rearrange
        RandomOrder = np.random.permutation(DatasT1wPatch.shape[0])
        DatasT1wPatch = DatasT1wPatch[RandomOrder,:,:,:,:]
        
        # Crop data to desired number of samples
        if args.samples :
            DatasT1wPatch = DatasT1wPatch[:args.samples ,...]
              
        # Writing to HDF5   
        hdf5name = args.out1[i]
        print '*) Writing to HDF5 file : ', hdf5name
        StartLocation = {'dat':(0,0,0,0,0)}
        CurrentDataLocation = store2hdf53Dgan(filename=hdf5name, 
                                           datas=DatasT1wPatch, 
                                           startloc=StartLocation, 
                                           chunksz=args.batch )
                                   
        # Reading HDF5 file                           
        with h5py.File(hdf5name,'r') as hf:
            udata = hf.get('data')
            print 'Shape of T1w patches:', udata.shape
            
        # Writing a text file which contains HDF5 file names 
        OutFile.write(hdf5name)
        OutFile.write('\n')
        
    # =================================================================================================
    # Read t2w reference images
    # Writing a text (.txt) file which contains HDF5 file names 
    OutFile = open(str(args.textT2), "w")
    
    for i in range(0,len(args.t2w)):
        
        T2wImage = args.t2w[i]
        print '================================================================'
        print 'Processing T2w image : ', T2wImage
        # Read NIFTI
        T2wImageNifti = sitk.ReadImage(T2wImage)
        
        # Get data from NIFTI
        T2wImage = np.swapaxes(sitk.GetArrayFromImage(T2wImageNifti),0,2).astype('float32')
        
        # Normalization by the max valeur
        MaxValue = np.max(T2wImage)
        NormalizedT2wImage = T2wImage/MaxValue

        # Shave region outside
        print 'Remove the region outside the brain with the value of ', args.thresholdvalue
        darkRegionValue = args.thresholdvalue
        darkRegionBox = np.where(NormalizedT2wImage>darkRegionValue)   
        border = ((np.min(darkRegionBox[0]),np.max(darkRegionBox[0])),
                  (np.min(darkRegionBox[1]),np.max(darkRegionBox[1])),
                  (np.min(darkRegionBox[2]),np.max(darkRegionBox[2])))     
        DatasT2wImage = NormalizedT2wImage[border[0][0]:border[0][1],
                                           border[1][0]:border[1][1],
                                           border[2][0]:border[2][1]]    
        
        # Extract 3D patches                              
        DatasT2wPatch = array_to_patches(DatasT2wImage, 
                                         patch_shape=(PatchSize,PatchSize,PatchSize), 
                                         extraction_step = args.stride , 
                                         normalization=False)
        print 'for patches of training phase.'        
    
        # Add channel axis !  
        DatasT2wPatch = DatasT2wPatch[:,np.newaxis,:,:,:]
                
        # Rearrange
        RandomOrder = np.random.permutation(DatasT2wPatch.shape[0])
        DatasT2wPatch = DatasT2wPatch[RandomOrder,:,:,:,:]
        
        # Crop data to desired number of samples
        if args.samples :
            DatasT2wPatch = DatasT2wPatch[:args.samples ,...]
            
        # =================================================================================================       
        # Writing to HDF5   
        hdf5name = args.out2[i]
        print '*) Writing to HDF5 file : ', hdf5name
        StartLocation = {'dat':(0,0,0,0,0)}
        CurrentDataLocation = store2hdf53Dgan(filename=hdf5name, 
                                           datas=DatasT2wPatch,
                                           startloc=StartLocation, 
                                           chunksz=args.batch )
                                   
        # Reading HDF5 file                           
        with h5py.File(hdf5name,'r') as hf:
            udata = hf.get('data')
            print 'Shape of T2w patches:', udata.shape
            
        # Writing a text file which contains HDF5 file names 
        OutFile.write(hdf5name)
        OutFile.write('\n')