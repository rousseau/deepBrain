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
from ast import literal_eval as make_tuple

import sys
sys.path.insert(0, './utils')
from utils3d import shave3D
from UnSynGAN import UnSynGAN

class UnSynGAN_test(object):
        def __init__(self,weights,T1toT2,
                     FirstDiscriminatorKernel=32,
                     FirstGeneratorKernel=16,
                     patch=128):
            self.patch = patch
            self.prediction = None
            self.UnSynGAN = UnSynGAN(ImageRow = patch, 
                                     ImageColumn = patch, 
                                     ImageDepth = patch,
                                     FirstDiscriminatorKernel = FirstDiscriminatorKernel, 
                                     FirstGeneratorKernel = FirstGeneratorKernel)
            self.GeneratorModel = self.UnSynGAN.generator_model()
            self.GeneratorModel.load_weights(weights, by_name=True)
            self.generator = self.UnSynGAN.generator()
            self.T1toT2 = T1toT2
            
        def test_by_patch(self,TestImage,step=2):  
            
            # Init temp
            Height,Width,Depth = np.shape(TestImage)
            TempSynImage = np.zeros_like(TestImage)
            WeightedImage = np.zeros_like(TestImage)
            if self.T1toT2 == 0:
                nclass = np.zeros([1,1])
            elif self.T1toT2 == 1:
                nclass = np.ones([1,1])
                    
            for idx in range(0,Height-self.patch+1,step):
                for idy in range(0,Width-self.patch+1,step):
                    for idz in range(0,Depth-self.patch+1,step):  
                        print '_', 
                        # Cropping image
                        TestPatch = TestImage[idx:idx+self.patch,idy:idy+self.patch,idz:idz+self.patch] 
                        ImageTensor = TestPatch.reshape(1,1,self.patch,self.patch,self.patch).astype(np.float32)
                        PredictPatch, _ =  self.generator.predict([ImageTensor,nclass], batch_size=1)
                        
                        # Adding
                        TempSynImage[idx:idx+self.patch,idy:idy+self.patch,idz:idz+self.patch] += PredictPatch[0,0,:,:,:]
                        WeightedImage[idx:idx+self.patch,idy:idy+self.patch,idz:idz+self.patch] += np.ones_like(PredictPatch[0,0,:,:,:])
                    
            # Weight sum of patches
            print 'Done !'
            EstimatedSynImage = TempSynImage/WeightedImage
            return EstimatedSynImage
    
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--test', help='Testing low-resolution image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-o', '--outputsr', help='Estimated high-resolution image filename (required)', type=str, action='append', required = True)
    parser.add_argument('-w', '--weights', help='Name of weight model (required)', type=str, required = True)
    parser.add_argument('-s', '--step', help='Stride for testing image (default=1)', type=int, default=1)
    parser.add_argument('--T1toT2', help='Desired synthetic image, if True T1toT2, if False T2toT1', type=str, default='True')
    parser.add_argument('-b', '--border', help='Border of interpolated image to remove (default=(50,50,0))', type=str, default='50,50,0')
   
    args = parser.parse_args()
    
    # Check number of test image and result name:
    if (len(args.outputsr) != len(args.test)):   
        raise AssertionError, 'Number of test images and result names should be matched !'   
            
    # Check border removing
    border = make_tuple(args.border)
    if np.isscalar(border):
        border = (border,border,border)
    else:
        if len(border)!=3:
            raise AssertionError, 'Not support this border !' 
        
    # Check T1w to T2w :
    if args.T1toT2 == 'True':
        T1toT2 = 1
        print 'Generating synthetic T2w images from T1w images'
    elif args.T1toT2 == 'False':
        T1toT2 = 0
        print 'Generating synthetic T1w images from T2w images'
    else:
        raise AssertionError, 'Not support this mode (Try True or False) !' 
        
    # Loading weights
    UnSynGAN_test = UnSynGAN_test(args.weights,T1toT2)
    
    for i in range(0,len(args.test)):
       
        # Read low-resolution image
        TestFile = args.test[i]
        print 'Processing testing image : ', TestFile 
        TestNifti = sitk.ReadImage(TestFile)
        TestImage = np.swapaxes(sitk.GetArrayFromImage(TestNifti),0,2).astype('float32')
        TestImageMinValue = float(np.min(TestImage))
        TestImageMaxValue = float(np.max(TestImage))
        TestImageNorm = TestImage/TestImageMaxValue 
        
        # Shave border
        ShavedImage = shave3D(TestImage, border)   

        # GAN 
        print "Testing : ",
        EstimatedImage = UnSynGAN_test.test_by_patch(ShavedImage,step=args.step)
        
        # Padding
        pad_border = [(idx,idx) for idx in border]
        PaddedEstimatedImage = np.pad(EstimatedImage,pad_border,'constant')
        
        # Synthetic image 
        EstimatedHRImageInverseNorm = PaddedEstimatedImage*TestImageMaxValue
        EstimatedHRImageInverseNorm[EstimatedHRImageInverseNorm <= TestImageMinValue] = TestImageMinValue    # Clear negative value
        OutputImage = sitk.GetImageFromArray(np.swapaxes(EstimatedHRImageInverseNorm,0,2))
        OutputImage.SetSpacing(TestNifti.GetSpacing())
        OutputImage.SetOrigin(TestNifti.GetOrigin())
        OutputImage.SetDirection(TestNifti.GetDirection())
        
        # Save result
        OutFile = args.outputsr[i]
        print 'SR image resust  : ', OutFile 
        sitk.WriteImage(OutputImage,OutFile)
