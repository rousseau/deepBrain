# Brain MRI super-resolution using deep 3D convolutional networks

### Table of Contents

1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Dependencies](#dependencies)
1. [Testing](#testing)
1. [Training](#training)

### 1. Introduction
This repository contains the SRCNN3D model described in the paper "Brain MRI super-resolution using deep 3D convolutional networks" (http://ieeexplore.ieee.org/abstract/document/7950500/).

### 2. Citation

If you use these models in your research, please cite:
```
 @inproceedings{pham2017brain,
      title={Brain MRI super-resolution using deep 3D convolutional networks},
      author={Pham, Chi-Hieu and Ducournau, Aur{\'e}lien and Fablet, Ronan and Rousseau, Fran{\c{c}}ois},
      booktitle={Biomedical Imaging (ISBI 2017), 2017 IEEE 14th International Symposium on},
      pages={197--200},
      year={2017},
      organization={IEEE}
  }
```
### 3. Dependencies

#### For reading and writing NIFTI data:
[SimpleITK](https://itk.org/Wiki/SimpleITK/GettingStarted)

#### For training
[Caffe](https://github.com/BVLC/caffe/)

h5py
#### For GPU testing:
[Lasagne](https://lasagne.readthedocs.io/en/latest/)

[cudnn](https://developer.nvidia.com/cudnn)

### 4. Testing

```
cd Test
python demo_SRCNN3D.py -t ($Dataset)/KKI2009-01-MPRAGE_LR.nii.gz -r KKI2009-01-MPRAGE_LR_SRCNN3D.nii.gz -m caffe_model/SRCNN3D_iter_470000.caffemodel -n caffe_model/SRCNN3D_deploy.prototxt -g True
```
t : testing LR image

r : SR image result

m : trained parameters of network stored in caffemodel file

n : deploy network

g : True (GPU mode)

Other arguments see : 
```
python demo_SRCNN3D.py -h
```

### 5. Training
#### Step 1 : Generating HDF5 files of training data and a text file of network protocol
```
cd Train
python generate_training.py -f ($Dataset)/KKI2009-33-MPRAGE.nii.gz -o hdf5/KKI2009-33-MPRAGE.hdf5 -f ($Dataset)/KKI2009-34-MPRAGE.nii.gz -o hdf5/KKI2009-34-MPRAGE.hdf5 -s 2,2,2 -s 3,3,3
```
f : HR reference image

o : HDF5 file which contains the patches of HR reference image

s : scale factors

Other arguments see : 
```
python generate_training.py -h
```

#### Step 2 : Creating a solver text file for Caffe
We can modify the solver text file at *model/SRCNN3D_solver.prototxt*

For further information about Caffe solver at [here](http://caffe.berkeleyvision.org/tutorial/solver.html)

**Or** using this function:
```
python generate_solver.py -l 0.0001 -s Adam
```
l : learning rate

s : optimization method

#### Step 3 : Training network using Caffe
```
mkdir caffe_model
caffe train --solver model/SRCNN3D_solver.prototxt
```

