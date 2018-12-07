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
from SRReCNN3D import SRReCNN3D
from utils3d import shave3D

class SRReCNN3D_test(object):
        def __init__(self):
            self.prediction = None
        
        def test(self,TestImage,weights,NetDepth,NetNumKernel,KernelSize,Residual):
            self.ImageRow = TestImage.shape[0]
            self.ImageColumn = TestImage.shape[1]
            self.ImageDepth = TestImage.shape[2]
            
            self.SRReCNN3D = SRReCNN3D(ImageRow =self.ImageRow, 
                                       ImageColumn = self.ImageColumn, 
                                       ImageDepth = self.ImageDepth,
                                       NetDepth = NetDepth,
                                       NetNumKernel = NetNumKernel,
                                       KernelSize = KernelSize,
                                       Residual = Residual)
            self.generator = self.SRReCNN3D.generator()
            self.generator.load_weights(weights, by_name=True)
                
            self.ImageTensor = TestImage.reshape(1, 1, self.ImageRow, self.ImageColumn, self.ImageDepth).astype(np.float32)
                        
            self.prediction =  self.generator.predict(self.ImageTensor, batch_size=1)
            return self.prediction[0,0,:,:,:]
    
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--test', help='Testing low-resolution image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-o', '--outputname', help='Estimated high-resolution image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-w', '--weights', help='Name of weight model (required)', type=str, required = True)
    parser.add_argument('-n', '--newhighres', help='Desired high resolution (default = (0.5,0.5,0.5))', type=str, default='0.5,0.5,0.5')
    parser.add_argument('-l', '--layers', help='Layer number of network (default = 20)', type=int, default=20)
    parser.add_argument('--order', help='Order of spline interpolation (default=3) ', type=int, default=3)
    parser.add_argument('-c', '--channel', help='Number of channels of training data (default=1)', type=int, default=1)
    parser.add_argument('-d', '--netdepth', help='Depth of network (default=20)', type=int, default=20)
    parser.add_argument('-k', '--numkernel', help='Number of filters of network (default=64)', type=int, default=64)
    parser.add_argument('-f', '--kernelsize', help='Filter size (default=3)', type=int, default=3)
    parser.add_argument('-r', '--residual', help='Using residual (Skip Connection) or None (default=True)', type=str, default='True')
    parser.add_argument('-b', '--border', help='Border of interpolated image to remove (default=(45,15,0))', type=str, default='45,15,0')
 
    args = parser.parse_args()
    
    # Check number of test image and result name:
    if len(args.outputname) != len(args.test):   
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
            
    # Check residual learning mode
    if args.residual == 'True':
        residual = True
    elif args.residual == 'False':
        residual = False
    else:
        raise AssertionError, 'Not support this residual mode. Try True or False !'     
        
    # Weights
    weights= args.weights
    
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
        
        # SRResidualCNN3D
        SRReCNN3D_test = SRReCNN3D_test()

        EstimatedHRImage = SRReCNN3D_test.test(ShavedInterpolatedImage,weights,
                                               NetDepth = args.netdepth, 
                                               NetNumKernel = args.numkernel, 
                                               KernelSize = args.kernelsize,
                                               Residual= residual)
        
        # Padding
        pad_border = [(idx,idx) for idx in border]
        PaddedEstimatedHRImage = np.pad(EstimatedHRImage,pad_border,'constant')
        
        
        # Save image 
        EstimatedHRImageInverseNorm = PaddedEstimatedHRImage*TestImageMaxValue
        EstimatedHRImageInverseNorm[EstimatedHRImageInverseNorm <= TestImageMinValue] = TestImageMinValue    # Clear negative value
        OutputImage = sitk.GetImageFromArray(np.swapaxes(EstimatedHRImageInverseNorm,0,2))
        OutputImage.SetSpacing(NewResolution)
        OutputImage.SetOrigin(TestNifti.GetOrigin())
        OutputImage.SetDirection(TestNifti.GetDirection())
        
        # Save result
        OutFile = args.outputname[i]
        print 'SR image resust  : ', OutFile 
        sitk.WriteImage(OutputImage,OutFile)
