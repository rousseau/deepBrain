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

import sys
sys.path.insert(0, './utils')
from utils3d import shave3D
from SegSRGAN import SegSRGAN

class SegSRGAN_test(object):
        def __init__(self,weights,patch=64):
            self.patch = patch
            self.prediction = None
            self.SegSRGAN = SegSRGAN(ImageRow = patch, 
                                     ImageColumn = patch, 
                                     ImageDepth = patch)
            self.GeneratorModel = self.SegSRGAN.generator_model()
            self.GeneratorModel.load_weights(weights, by_name=True)
            self.generator = self.SegSRGAN.generator()
            
        def test_by_patch(self,TestImage,step=1):  
            
            # Init temp
            Height,Width,Depth = np.shape(TestImage)
            TempHRImage = np.zeros_like(TestImage)
            TempSeg = np.zeros_like(TestImage)
            WeightedImage = np.zeros_like(TestImage)
    
            for idx in range(0,Height-self.patch+1,step):
                for idy in range(0,Width-self.patch+1,step):
                    for idz in range(0,Depth-self.patch+1,step):  
                        print '_', 
                        # Cropping image
                        TestPatch = TestImage[idx:idx+self.patch,idy:idy+self.patch,idz:idz+self.patch] 
                        ImageTensor = TestPatch.reshape(1,1,self.patch,self.patch,self.patch).astype(np.float32)
                        PredictPatch =  self.generator.predict(ImageTensor, batch_size=1)
                        
                        # Adding
                        TempHRImage[idx:idx+self.patch,idy:idy+self.patch,idz:idz+self.patch] += PredictPatch[0,0,:,:,:]
                        TempSeg [idx:idx+self.patch,idy:idy+self.patch,idz:idz+self.patch] += PredictPatch[0,1,:,:,:]
                        WeightedImage[idx:idx+self.patch,idy:idy+self.patch,idz:idz+self.patch] += np.ones_like(PredictPatch[0,0,:,:,:])
                    
            # Weight sum of patches
            print 'Done !'
            EstimatedHR = TempHRImage/WeightedImage
            EstimatedSegmentation = TempSeg/WeightedImage
            return (EstimatedHR,EstimatedSegmentation)
    
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--test', help='Testing low-resolution image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-o', '--outputsr', help='Estimated high-resolution image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-c', '--cortexseg', help='Estimated cortex segmentation filename (required)', type=str, action='append', required = True)
    parser.add_argument('-w', '--weights', help='Name of weight model (required)', type=str, required = True)
    parser.add_argument('-s', '--step', help='Stride for testing image (default=1)', type=int, default=1)
    parser.add_argument('-n', '--newhighres', help='Desired high resolution (default = (0.5,0.5,0.5))', type=str, default='0.5,0.5,0.5')
    parser.add_argument('--order', help='Order of spline interpolation (default=3) ', type=int, default=3)
    parser.add_argument('-b', '--border', help='Border of interpolated image to remove (default=(45,15,0))', type=str, default='45,15,0')
 
    args = parser.parse_args()
    
    # Check number of test image and result name:
    if (len(args.outputsr) != len(args.test)) or (len(args.cortexseg) != len(args.test)):   
        raise AssertionError, 'Number of test images and result names should be matched !'   

     # Check resolution
    NewResolution = make_tuple(args.newhighres)
    if np.isscalar(NewResolution):
        NewResolution = (NewResolution,NewResolution,NewResolution)
    else:
        if len(NewResolution)!=3:
            raise AssertionError, 'Not support this resolution !'  
            
    # Check border removing
    border = make_tuple(args.border)
    if np.isscalar(border):
        border = (border,border,border)
    else:
        if len(border)!=3:
            raise AssertionError, 'Not support this border !' 
        
    # Loading weights
    SegSRGAN_test = SegSRGAN_test(args.weights)
    
    for i in range(0,len(args.test)):
       
        # Read low-resolution image
        TestFile = args.test[i]
        print 'Processing testing image : ', TestFile 
        TestNifti = sitk.ReadImage(TestFile)
        TestImage = np.swapaxes(sitk.GetArrayFromImage(TestNifti),0,2).astype('float32')
        TestImageMinValue = float(np.min(TestImage))
        TestImageMaxValue = float(np.max(TestImage))
        TestImageNorm = TestImage/TestImageMaxValue 
        
        # Check scale factor type
        UpScale = tuple(itema/itemb for itema,itemb in zip(TestNifti.GetSpacing(),NewResolution)) 
        
        # spline interpolation 
        InterpolatedImage = scipy.ndimage.zoom(TestImageNorm, 
                                               zoom = UpScale,
                                               order = args.order)  
        # Shave border
        ShavedInterpolatedImage = shave3D(InterpolatedImage, border)   
        
        # GAN 
        print "Testing : ",
        EstimatedHRImage, EstimatedCortex  = SegSRGAN_test.test_by_patch(ShavedInterpolatedImage,step=args.step)
        
        # Padding
        pad_border = [(idx,idx) for idx in border]
        PaddedEstimatedHRImage = np.pad(EstimatedHRImage,pad_border,'constant')
        
        # SR image 
        EstimatedHRImageInverseNorm = PaddedEstimatedHRImage*TestImageMaxValue
        EstimatedHRImageInverseNorm[EstimatedHRImageInverseNorm <= TestImageMinValue] = TestImageMinValue    # Clear negative value
        OutputImage = sitk.GetImageFromArray(np.swapaxes(EstimatedHRImageInverseNorm,0,2))
        OutputImage.SetSpacing(NewResolution)
        OutputImage.SetOrigin(TestNifti.GetOrigin())
        OutputImage.SetDirection(TestNifti.GetDirection())
        
        # Cortex segmentation
        OutputCortex = sitk.GetImageFromArray(np.swapaxes(EstimatedCortex,0,2))
        OutputCortex.SetSpacing(NewResolution)
        OutputCortex.SetOrigin(TestNifti.GetOrigin())
        OutputCortex.SetDirection(TestNifti.GetDirection())
        
        # Save result
        OutFile = args.outputsr[i]
        print 'SR image resust  : ', OutFile 
        sitk.WriteImage(OutputImage,OutFile)
        
        OutFile = args.cortexseg[i]
        print 'Cortex segmentation resust  : ', OutFile 
        sitk.WriteImage(OutputCortex,OutFile)
