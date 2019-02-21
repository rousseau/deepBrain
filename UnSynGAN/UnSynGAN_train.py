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
from store2hdf5 import ProcessingTrainingSetFromTextFiles
from UnSynGAN import UnSynGAN

class UnSynGAN_train(object):
    def __init__(self,
                 T1TrainingText = 'model/trainT1.txt', T2TrainingText = 'model/trainT2.txt', 
                 patch=128, 
                 FirstDiscriminatorKernel = 32, FirstGeneratorKernel = 16,
                 lamb_rec = 1000, lamb_adv = 1, lamb_gp = 10, lamb_tv = 0.001,
                 lr_DisModel = 0.0001, lr_GenModel = 0.0001):
        
        self.UnSynGAN = UnSynGAN(ImageRow=patch, ImageColumn=patch, ImageDepth=patch, 
                                 FirstDiscriminatorKernel = FirstDiscriminatorKernel, 
                                 FirstGeneratorKernel = FirstGeneratorKernel,
                                 lamb_rec = lamb_rec, lamb_adv = lamb_adv, lamb_gp = lamb_gp, lamb_tv = lamb_tv,
                                 lr_DisModel = lr_DisModel, lr_GenModel = lr_GenModel)
        self.generator = self.UnSynGAN.generator()
        self.T1TrainingText = T1TrainingText
        self.T2TrainingText = T2TrainingText
        self.DiscriminatorModel = self.UnSynGAN.discriminator_model()
        self.GeneratorModel = self.UnSynGAN.generator_model()
        
    def train(self, 
              TrainingEpoch=200, BatchSize=2, SnapshotEpoch=1, InitializeEpoch=1, NumCritic=5, 
              resuming = None):
        
        snapshot_prefix='weights/UnSynGAN_epoch'
        if os.path.exists('weights') is False:
            os.makedirs('weights')
        
        # Initialization Parameters
        real = -np.ones([BatchSize, 1], dtype=np.float32)
        fake = -real
        dummy = np.zeros([BatchSize, 1], dtype=np.float32)   
        classT2 = np.ones([BatchSize,1])
        classT1 = np.zeros([BatchSize,1])
        
        # Data processing
        T1wTrainingSet = ProcessingTrainingSetFromTextFiles(self.T1TrainingText,BatchSize, InputName='data')
        T2wTrainingSet = ProcessingTrainingSetFromTextFiles(self.T2TrainingText,BatchSize, InputName='data')
        iterationPerEpoch = T1wTrainingSet.iterationPerEpoch

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
                iteration = (InitializeEpoch-1)*iterationPerEpoch
        
        # Training phase
        for EpochIndex in range(InitializeEpoch,TrainingEpoch+1):  
            print "Processing epoch : " + str(EpochIndex)
            for iters in range(0,iterationPerEpoch):
                iteration += 1
                
                # Training discriminator
                for cidx in range(NumCritic):
                    # Loading data randomly
                    randNumber = int(np.random.randint(0,iterationPerEpoch,1))    
                    indice = randNumber*BatchSize
                    realT1w = T1wTrainingSet.load_batch(indice)
                    realT2w = T2wTrainingSet.load_batch(indice)
                    
                    # Generating fake and interpolation images
                    fakeT2w, EmClassT2 = self.generator.predict([realT1w,classT2])
                    fakeT1w, EmClassT1 = self.generator.predict([realT2w,classT1])
                    epsilon = np.random.uniform(0, 1, size=(BatchSize,1,1,1,1))
                    interpT1w = epsilon*fakeT2w + (1-epsilon)*realT1w
                    interpT2w = epsilon*fakeT1w + (1-epsilon)*realT2w
                    
                    # Training
                    dis_loss1 = self.DiscriminatorModel.train_on_batch([realT1w,fakeT1w,interpT1w,EmClassT1],
                                                                       [real,fake,dummy])
                    print "Update T1w "+ str(cidx) + ": [D loss: "+str(dis_loss1)+"]"  
                    dis_loss2 = self.DiscriminatorModel.train_on_batch([realT2w,fakeT2w,interpT2w,EmClassT2],
                                                                       [real,fake,dummy])
                    print "Update T2w "+ str(cidx) + ": [D loss: "+str(dis_loss2)+"]"  
                    
                # Training generator
                # Loading data
                indice4Generator = iters*BatchSize  
                T1images = T1wTrainingSet.load_batch(indice4Generator)
                T2images = T2wTrainingSet.load_batch(indice4Generator)

                # Training                                      
                adv_loss1 = self.GeneratorModel.train_on_batch([T1images,classT2,classT1],  
                                                               [real,T1images,T1images])
                adv_loss2 = self.GeneratorModel.train_on_batch([T2images,classT1,classT2],  
                                                               [real,T2images,T2images])
                print "Iter "+ str(iteration) + " T1w [G loss: " + str(adv_loss1) + "]"  
                print "Iter "+ str(iteration) + " T2w [G loss: " + str(adv_loss2) + "]"  
                
            if (EpochIndex)%SnapshotEpoch==0:
                # Save weights:
                self.GeneratorModel.save_weights(snapshot_prefix + '_' + str(EpochIndex))
                print ("Snapshot :" + snapshot_prefix + '_' + str(EpochIndex))                         
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t1', '--textT1', help='Name of a text (.txt) file which contains HDF5 file names of T1w images (default: trainT1.txt)', type=str, default='trainT1.txt')
    parser.add_argument('-t2', '--textT2', help='Name of a text (.txt) file which contains HDF5 file names of T2w images  (default: trainT2.txt)', type=str, default='trainT2.txt')
    parser.add_argument('-e', '--epoch', help='Number of training epochs (default=200)', type=int, default=200)
    parser.add_argument('-b', '--batchsize', help='Number of batch (default=2)', type=int, default=1)
    parser.add_argument('-s', '--snapshot', help='Snapshot Epoch (default=1)', type=int, default=1)
    parser.add_argument('-i', '--initepoch', help='Init Epoch (default=1)', type=int, default=1)
    parser.add_argument('-w', '--weights', help='Name of the pretrained HDF5 weight file (default: None)', type=str, default=None)
    parser.add_argument('--kernelgen', help='Number of filters of the first layer of generator (default=16)', type=int, default=16)
    parser.add_argument('--kerneldis', help='Number of filters of the first layer of discriminator (default=32)', type=int, default=32)
    parser.add_argument('--lrgen', help='Learning rate of generator (default=0.0001)', type=int, default=0.0001)
    parser.add_argument('--lrdis', help='Learning rate of discriminator (default=0.0001)', type=int, default=0.0001)
    parser.add_argument('--lambrec', help='Lambda of reconstruction loss (default=1000)', type=int, default=5000)
    parser.add_argument('--lambadv', help='Lambda of adversarial loss (default=1)', type=int, default=1)
    parser.add_argument('--lambgp', help='Lambda of gradient penalty loss (default=10)', type=int, default=10)
    parser.add_argument('--lambtv', help='Lambda of total variation loss (default=0.001)', type=int, default=0.001)
    parser.add_argument('--numcritic', help='Number of training time for discriminator (default=5) ', type=int, default=5)
    
    args = parser.parse_args()
    UnSynGAN_train = UnSynGAN_train(T1TrainingText = args.textT1,  T2TrainingText = args.textT2, 
                              FirstDiscriminatorKernel = args.kerneldis, FirstGeneratorKernel = args.kernelgen,
                              lamb_rec = args.lambrec, lamb_adv = args.lambadv, lamb_gp = args.lambgp, lamb_tv = args.lambtv, 
                              lr_DisModel = args.lrdis, lr_GenModel = args.lrgen)
    UnSynGAN_train.train(TrainingEpoch=args.epoch, BatchSize=args.batchsize, 
                         SnapshotEpoch=args.snapshot, InitializeEpoch = args.initepoch,
                         NumCritic = args.numcritic,
                         resuming = args.weights)
