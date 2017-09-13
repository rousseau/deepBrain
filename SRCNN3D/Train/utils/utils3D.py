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

def shave3D(image,border):
    '''
    Remove border of an image
    Note: Array of Python in interval [i:j] will affect from i to j-1
    -----
    image : 3d array
    border : tuple, indicates border is removed
    Example : border = (10,10,10)
    
    '''
    if np.isscalar(border):
        image = image[border:image.shape[0]-border,border:image.shape[1]-border, border:image.shape[2]-border]
    else:
        image = image[border[0]:image.shape[0]-border[0],border[1]:image.shape[1]-border[1], border[2]:image.shape[2]-border[2]]
    return image

def imadjust3D(image, newRange = None):
    """
        More detail about formula : https://en.wikipedia.org/wiki/Normalization_(image_processing)
        ----
        image : 3d array
        newRange : new range of value
        Example : newRange = [0,1]
    """
    Min = np.min(image)
    Max = np.max(image)
    newMin = newRange[0]
    newMax = newRange[1]
    temp = (newMax - newMin) / float(Max - Min)
    image = ((image - Min) * temp + newMin)
    return image 

def modcrop3D(img, modulo):
    import math
    img = img[0:int(img.shape[0] - math.fmod(img.shape[0], modulo[0])), 
              0:int(img.shape[1] - math.fmod(img.shape[1], modulo[1])), 
              0:int(img.shape[2] - math.fmod(img.shape[2], modulo[2]))]
    return img
