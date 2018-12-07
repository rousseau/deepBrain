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
import argparse
import os
import sys
sys.path.insert(0, './utils')
from store2hdf5 import ProcessingTrainingSet
from SegSRGAN import SegSRGAN

class SegSRGAN_train(object):
    def __init__(self,
                 TrainingText = 'model/train.txt', 
                 patch=64, 
                 FirstDiscriminatorKernel = 32, FirstGeneratorKernel = 16,
                 lamb_rec = 1, lamb_adv = 0.001, lamb_gp = 10, 
                 lr_DisModel = 0.0001, lr_GenModel = 0.0001):
        
        self.SegSRGAN = SegSRGAN(ImageRow=patch, ImageColumn=patch, ImageDepth=patch, 
                                 FirstDiscriminatorKernel = FirstDiscriminatorKernel, 
                                 FirstGeneratorKernel = FirstGeneratorKernel,
                                 lamb_rec = lamb_rec, lamb_adv = lamb_adv, lamb_gp = lamb_gp, 
                                 lr_DisModel = lr_DisModel, lr_GenModel = lr_GenModel)
        self.generator = self.SegSRGAN.generator()
        self.TrainingText = TrainingText
        self.DiscriminatorModel = self.SegSRGAN.discriminator_model()
        self.GeneratorModel = self.SegSRGAN.generator_model()
        
    def train(self, 
              TrainingEpoch=200, BatchSize=16, SnapshotEpoch=1, InitializeEpoch=1, NumCritic=5, 
              resuming = None):
        
        snapshot_prefix='weights/SegSRGAN_epoch'
        if os.path.exists('weights') is False:
            os.makedirs('weights')
        
        # Initialization Parameters
        real = -np.ones([BatchSize, 1], dtype=np.float32)
        fake = -real
        dummy = np.zeros([BatchSize, 1], dtype=np.float32)   
        
        # Data processing
        TrainingSet = ProcessingTrainingSet(self.TrainingText,BatchSize, InputName='data', LabelName = 'label')

        # Resuming
        if InitializeEpoch==1:
            iteration = 0
            if resuming is None:
                print "Training from scratch"
            else:
                print "Training from the pretrained model (names of layers must be identical): ", resuming
                self.GeneratorModel.load_weights(resuming, by_name=True)
        
        elif InitializeEpoch <1:
            raise AssertionError, 'Resumming needs a positive epoch'
        else:
            if resuming is None:
                raise AssertionError, 'We need pretrained weights'
            else:
                print 'Continue training from : ', resuming
                self.GeneratorModel.load_weights(resuming, by_name=True)
                iteration = (InitializeEpoch-1)*TrainingSet.iterationPerEpoch
        
        # Training phase
        for EpochIndex in range(InitializeEpoch,TrainingEpoch+1):  
            print "Processing epoch : " + str(EpochIndex)
            for iters in range(0,TrainingSet.iterationPerEpoch):
                iteration += 1
                
                # Training discriminator
                for cidx in range(NumCritic):
                    # Loading data randomly
                    randomNumber = int(np.random.randint(0,TrainingSet.iterationPerEpoch,1))
                    DataIndice = randomNumber*BatchSize
                    train_input = TrainingSet.Datas[DataIndice:DataIndice+BatchSize,:,:,:,:]  
                    train_output = TrainingSet.Labels[DataIndice:DataIndice+BatchSize,:,:,:,:]     
                    
                    # Generating fake and interpolation images
                    fake_images = self.generator.predict(train_input)
                    epsilon = np.random.uniform(0, 1, size=(BatchSize,2,1,1,1))
                    interpolation = epsilon*train_output + (1-epsilon)*fake_images

                    # Training
                    dis_loss = self.DiscriminatorModel.train_on_batch([train_output,fake_images,interpolation],
                                                                       [real,fake,dummy])
                    print "Update "+ str(cidx) + ": [D loss: "+str(dis_loss)+"]"  
                    
                # Training generator
                # Loading data
                DataIndice = iters*BatchSize                
                train_input_gen = TrainingSet.Datas[DataIndice:DataIndice+BatchSize,:,:,:,:]  
                train_output_gen = TrainingSet.Labels[DataIndice:DataIndice+BatchSize,:,:,:,:]  

                # Training                                      
                gen_loss = self.GeneratorModel.train_on_batch([train_input_gen], 
                                                               [real,train_output_gen])
                print "Iter "+ str(iteration) + " [A loss: " + str(gen_loss) + "]"  
                
            if (EpochIndex)%SnapshotEpoch==0:
                # Save weights:
                self.GeneratorModel.save_weights(snapshot_prefix + '_' + str(EpochIndex))
                print ("Snapshot :" + snapshot_prefix + '_' + str(EpochIndex))
                          


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--text', help='Name of a text (.txt) file which contains HDF5 file names (default: train.txt)', type=str, default='train.txt')
    parser.add_argument('-e', '--epoch', help='Number of training epochs (default=200)', type=int, default=200)
    parser.add_argument('-b', '--batchsize', help='Number of batch (default=16)', type=int, default=16)
    parser.add_argument('-s', '--snapshot', help='Snapshot Epoch (default=1)', type=int, default=1)
    parser.add_argument('-i', '--initepoch', help='Init Epoch (default=1)', type=int, default=1)
    parser.add_argument('-w', '--weights', help='Name of the pretrained HDF5 weight file (default: None)', type=str, default=None)
    parser.add_argument('--kernelgen', help='Number of filters of the first layer of generator (default=16)', type=int, default=16)
    parser.add_argument('--kerneldis', help='Number of filters of the first layer of discriminator (default=32)', type=int, default=32)
    parser.add_argument('--lrgen', help='Learning rate of generator (default=0.0001)', type=int, default=0.0001)
    parser.add_argument('--lrdis', help='Learning rate of discriminator (default=0.0001)', type=int, default=0.0001)
    parser.add_argument('--lambrec', help='Lambda of reconstruction loss (default=1)', type=int, default=1)
    parser.add_argument('--lambadv', help='Lambda of adversarial loss (default=0.001)', type=int, default=0.001)
    parser.add_argument('--lambgp', help='Lambda of gradien penalty loss (default=10)', type=int, default=10)
    parser.add_argument('--numcritic', help='Number of training time for discriminator (default=5) ', type=int, default=5)
    
    args = parser.parse_args()
    SegSRGAN_train = SegSRGAN_train(TrainingText = args.text,  
                                    FirstDiscriminatorKernel = args.kerneldis, FirstGeneratorKernel = args.kernelgen,
                                    lamb_rec = args.lambrec, lamb_adv = args.lambadv, lamb_gp = args.lambgp, 
                                    lr_DisModel = args.lrdis, lr_GenModel = args.lrgen)
    SegSRGAN_train.train(TrainingEpoch=args.epoch, BatchSize=args.batchsize, 
                         SnapshotEpoch=args.snapshot, InitializeEpoch = args.initepoch,
                         NumCritic = args.numcritic,
                         resuming = args.weights)
