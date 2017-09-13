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

import os
import h5py
import numpy as np
from operator import add

def store2hdf53D(filename, datas, labels, create=True, startloc=None, chunksz=256) :
    '''
    Store patches in format HDF5
    Adapted from 2D stor2hdf5 of Matcaffe (Caffe for matlab)
    Caffe supports data's form : N*C*H*W*D (numberOfBatches,channels,heigh,width,depth) 
    
    ----------
     datas, labels : 5D array, 
         formed N*C*H*W*D matrix of images should be normalized (e.g. to lie between 0 and 1) beforehand
     create :  True or False (default: true)
         specifies whether to create file newly or to append to previously created file, 
         useful to store information in batches when a dataset is too big to be held in memory  
     startloc : (point at which to start writing data). 
         By default, 
         if create=1 (create mode), startloc.data=[1, 1, 1, 1, 1], and startloc.lab=[1, 1, 1, 1, 1]; 
         if create=0 (append mode), startloc.data=[K+1, 1, 1, 1, 1], and startloc.lab = [K+1, 1, 1, 1, 1]; 
         where K is the current number of samples stored in the HDF
     chunksz : integer  (default: 256)
         (used only in create mode):
         specifies number of samples to be stored per chunk (see HDF5 documentation on chunking) 
         for creating HDF5 files with unbounded maximum size - TLDR; 
         higher chunk sizes allow faster read-write operations                 
                     
    '''
    # verify that format is right
    dat_dims = datas.shape
    lab_dims = labels.shape
    num_samples = dat_dims[0]
    if (num_samples != lab_dims[0]):
        raise AssertionError, 'Number of samples should be matched between data and labels'    
            
    # Check Create mode and Startloc    
    if create == True:
        if os.path.isfile(filename):
            print 'Warning: replacing existing file', filename, '\n'
            os.remove(filename)
           
        file = h5py.File(filename, "w")
        dset = file.create_dataset("data", shape=dat_dims , dtype='float32', maxshape=(None,)+tuple(dat_dims[1:]), chunks=(np.long(chunksz),)+tuple(dat_dims[1:]))
        lset = file.create_dataset("label", shape=lab_dims , dtype='float32', maxshape=(None,)+tuple(lab_dims[1:]), chunks=(np.long(chunksz),)+tuple(lab_dims[1:]))
        dset[...] = datas
        lset[...] = labels
        if  startloc == None:
            startloc = {'dat':0 , 'lab':0}
            startloc['dat']= (0,) + tuple(np.zeros(len(dat_dims)-1,dtype='int'))
            startloc['lab']= (0,) + tuple(np.zeros(len(lab_dims)-1,dtype='int'))           
    
    else:   # append mode
        if  startloc == None:
            file = h5py.File(filename, "r")
            prev_dat_sz = file[file.keys()[0]].shape
            prev_lab_sz = file[file.keys()[1]].shape
            if ( prev_dat_sz[1:] != dat_dims[1:] ):
                raise AssertionError, 'Data dimensions must match existing dimensions in dataset'
            if ( prev_lab_sz[1:] != lab_dims[1:] ):
                raise AssertionError, 'Label dimensions must match existing dimensions in dataset'    
            startloc = {'dat':0 , 'lab':0}
            startloc['dat']= (prev_dat_sz[0],) + tuple(np.zeros(len(dat_dims)-1,dtype='int'))
            startloc['lab']= (prev_lab_sz[0],) + tuple(np.zeros(len(lab_dims)-1,dtype='int'))
            
    # Writing data
    if (datas.size) or (labels.size):
        file = h5py.File(filename, "r+")
        dset = file['/data']
        lset = file['/label']
        dset.resize(map(add, dat_dims,startloc['dat']))
        lset.resize(map(add, lab_dims,startloc['lab'])) 
        dset[startloc['dat'][0]:startloc['dat'][0]+dat_dims[0],:,:,:,:] = datas
        lset[startloc['lab'][0]:startloc['lab'][0]+lab_dims[0],:,:,:,:] = labels      
    else:    
        assert 'store2hdf5 need datas'
    curr_dat_sz=file[file.keys()[0]].shape
    curr_lab_sz=file[file.keys()[1]].shape
    file.close()   
    return [curr_dat_sz,curr_lab_sz]
