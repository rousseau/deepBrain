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

def store2hdf53Dgan(filename, datas, create=True, startloc=None, chunksz=1000) :
    # Caffe supports data's form : [numberOfBatches,channels,heigh,width,depth] 
    dat_dims = datas.shape
             
    # Check Create mode and Startloc    
    if create == True:
        if os.path.isfile(filename):
            print 'Warning: replacing existing file', filename, '\n'
            os.remove(filename)
           
        file = h5py.File(filename, "w")
        dset = file.create_dataset("data", shape=dat_dims , dtype='float32', maxshape=(None,)+tuple(dat_dims[1:]), chunks=(np.long(chunksz),)+tuple(dat_dims[1:]))
        dset[...] = datas
        if  startloc == None:
            startloc = {'dat':0}
            startloc['dat']= (0,) + tuple(np.zeros(len(dat_dims)-1,dtype='int'))
      
    else:   # append mode
        if  startloc == None:
            file = h5py.File(filename, "r")
            prev_dat_sz = file[file.keys()[0]].shape
            if ( prev_dat_sz[1:] != dat_dims[1:] ):
                raise AssertionError, 'Data dimensions must match existing dimensions in dataset'
            startloc = {'dat':0}
            startloc['dat']= (prev_dat_sz[0],) + tuple(np.zeros(len(dat_dims)-1,dtype='int'))
            
    # Writing data
    if (datas.size):
        file = h5py.File(filename, "r+")
        dset = file['/data']
        dset.resize(map(add, dat_dims,startloc['dat']))
        dset[startloc['dat'][0]:startloc['dat'][0]+dat_dims[0],:,:,:,:] = datas    
    else:    
        assert 'store2hdf5 need datas'
    curr_dat_sz=file[file.keys()[0]].shape
    file.close()   
    return [curr_dat_sz]

class ProcessingTrainingSetFromTextFiles(object):  
    def __init__(self, TrainingFilesText, batchSize, InputName = 'data'):  
        self.iterationPerEpoch = 0
        self.batchSize = batchSize
        self.locationFile = []
        self.InputName = InputName
        self.dataLength = 0
                
        # Get set of files
        with open(TrainingFilesText , "r") as TrainingFiles:
            string = TrainingFiles.read()
        self.FileSet = string.split()
        
        # Get location of each file in set and number of iteration each epoch
        for FileIndex in range(len(self.FileSet)):
            with h5py.File(self.FileSet[FileIndex],'r') as hf:
            # Reading all data
                if InputName in hf.keys():
                    datas = np.array(hf.get(InputName))
                    self.dataLength = self.dataLength+ datas.shape[0]
                else:
                    raise AssertionError, 'Key name ' + InputName + 'does not exist !'
             
            if FileIndex == 0:
                self.locationFile.append(datas.shape[0]) 
            else:
                self.locationFile.append(self.locationFile[FileIndex-1]+datas.shape[0])
        self.iterationPerEpoch = self.dataLength/ self.batchSize 
         
    def load_batch(self, batchIndexInSet = 0):
        # Find n-th file   
        FileIndex = 0
        while self.locationFile[FileIndex] <= batchIndexInSet: FileIndex +=1
        
        # Compute the index of batch in n-th file
        batchIndexInFile = batchIndexInSet 
        if FileIndex > 0 : batchIndexInFile = batchIndexInFile - self.locationFile[FileIndex-1]

        # Extract patch from n-th HDF5 file
        with h5py.File(self.FileSet[FileIndex],'r') as hf:
            # Reading data
            return  np.array(hf.get(self.InputName))[batchIndexInFile:batchIndexInFile+self.batchSize,:,:,:,:] 