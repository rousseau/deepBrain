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

import argparse
import os
import sys
sys.path.insert(0, './utils')
from store2hdf5 import ProcessingTrainingSet
from SRReCNN3D import SRReCNN3D

class SRReCNN3D_train(object):
    def __init__(self,
                 TrainingText = 'model/train.txt', 
                 patch=25, channel=1, NetDepth = 20, NetNumKernel = 64, KernelSize = 3, 
                 LearningRate = 0.0001, Residual= True):
        
        self.SRReCNN3D = SRReCNN3D(ImageRow = patch, ImageColumn = patch,
                                   ImageDepth = patch, channel = channel, 
                                   NetDepth = NetDepth, NetNumKernel = NetNumKernel, 
                                   KernelSize = KernelSize, LearningRate = LearningRate,
                                   Residual= Residual)
        self.generator = self.SRReCNN3D.generator()
        self.TrainingText = TrainingText
        
    def train(self, 
              TrainingEpoch=20, BatchSize=1, SnapshotEpoch=1, InitializeEpoch=1, 
              resuming = None):
        
        snapshot_prefix='weights/SRReCNN3D_epoch'
        if os.path.exists('weights') is False:
            os.makedirs('weights')
            
        # Data processing
        TrainingSet = ProcessingTrainingSet(self.TrainingText,BatchSize, InputName='data', LabelName = 'label')

        # Resuming
        if InitializeEpoch==1:
            iteration = 0
            if resuming is None:
                print "Training from scratch"
            else:
                print "Training from the pretrained model (names of layers must be identical): ", resuming
                self.generator.load_weights(resuming, by_name=True)
        
                
        elif InitializeEpoch <1:
            raise AssertionError, 'Resumming needs a positive epoch'
        else:
            if resuming is None:
                raise AssertionError, 'We need pretrained weights'
            else:
                print 'Continue training from : ', resuming
                self.generator.load_weights(resuming, by_name=True)
                iteration = (InitializeEpoch-1)*TrainingSet.iterationPerEpoch
            
        for EpochIndex in range(InitializeEpoch,TrainingEpoch+1):  
            print "Processing epoch : " + str(EpochIndex)
            for iters in range(0,TrainingSet.iterationPerEpoch):
                iteration += 1
                DataIndice = iters*BatchSize
                train_input = TrainingSet.Datas[DataIndice:DataIndice+BatchSize,:,:,:,:]  
                train_output = TrainingSet.Labels[DataIndice:DataIndice+BatchSize,:,:,:,:]                                  
                dis_loss = self.generator.train_on_batch(train_input, train_output)
                log_mesg = "Iteration %d: [G loss: %f]" % (iteration, dis_loss)
                print(log_mesg)
                
            if (EpochIndex)%SnapshotEpoch==0:
                # Save weights:
                self.generator.save_weights(snapshot_prefix + '_' + str(EpochIndex))
                print ("Snapshot :" + snapshot_prefix + '_' + str(EpochIndex))
                          


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--text', help='Name of a text (.txt) file which contains HDF5 file names (default: train.txt)', type=str, default='train.txt')
    parser.add_argument('-e', '--epoch', help='Number of training epochs (default=20)', type=int, default=20)
    parser.add_argument('-b', '--batchsize', help='Number of batch (default=64)', type=int, default=64)
    parser.add_argument('-s', '--snapshot', help='Snapshot Epoch (default=1)', type=int, default=1)
    parser.add_argument('-i', '--initepoch', help='Init Epoch (default=1)', type=int, default=1)
    parser.add_argument('-w', '--weights', help='Name of the pretrained HDF5 weight file (default: None)', type=str, default=None)
    parser.add_argument('-p', '--patch', help='Patch size, must be equal to training patch size (default=25)', type=int, default=25)
    parser.add_argument('-c', '--channel', help='Number of channels of training data (default=1)', type=int, default=1)
    parser.add_argument('-d', '--netdepth', help='Depth of network (default=20)', type=int, default=20)
    parser.add_argument('-k', '--numkernel', help='Number of filters of network (default=64)', type=int, default=64)
    parser.add_argument('-f', '--kernelsize', help='Filter size (default=3)', type=int, default=3)
    parser.add_argument('-r', '--residual', help='Using residual learning or None (default=True)', type=str, default='True')
    parser.add_argument('-l', '--learningrate', help='Learning rate (default=0.0001)', type=int, default=0.0001)
    
    args = parser.parse_args()
    
    # Check residual learning mode
    if args.residual == 'True':
        residual = True
    elif args.residual == 'False':
        residual = False
    else:
        raise AssertionError, 'Not support this residual mode. Try True or False !' 

    SRReCNN3D_train = SRReCNN3D_train(TrainingText = args.text,  
                                      patch=args.patch, channel=args.channel, NetDepth = args.netdepth, 
                                      NetNumKernel = args.numkernel, KernelSize = args.kernelsize,
                                      LearningRate = args.learningrate,  Residual= residual)
    SRReCNN3D_train.train(TrainingEpoch=args.epoch, BatchSize=args.batchsize, 
                          SnapshotEpoch=args.snapshot, InitializeEpoch = args.initepoch,
                          resuming = args.weights)
