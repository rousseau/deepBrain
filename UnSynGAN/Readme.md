# UnSynGAN (In progress)


### Table of Contents
0. [Introduction](#introduction)
1. [Citation](#citation)
1. [Dependencies](#dependencies)
1. [UnSynGAN](#UnSynGAN)

### 0. Introduction
This repository contains unsupervised MRI tissue contrast synthesis using generative adversarial networks

### 1. Citation
My thesis : Deep learning for image super-resolution and segmentation.
(Apprentisage profond pour la super-résolution et la segmentation d'image)
http://www.theses.fr/s201653

### 2. Dependencies

#### For reading and writing NIFTI data:
[SimpleITK](https://itk.org/Wiki/SimpleITK/GettingStarted)

#### For training
[Keras](https://keras.io/)

h5py

### 3. UnSynGAN

#### a) Testing

Testing

```
python UnSynGAN_test.py -t /home/chpham/these/data/NAMIC/NAMIC_T1T2/masked/T1/test/01011-t1w_masked.nii.gz -o 01011-t1w_masked_toT2.nii.gz.nii.gz -w weights/UnSynGAN --T1toT2 True
```
t : testing image

o : output SR result

w : trained weights

T1toT2 : if True (string): generating synthetic T2w images from T1w images, if False (string): generating synthetic T1w images from T2w images

Other arguments see : 
```
python UnSynGAN_test.py 
```

#### b) Training
##### Step 1 : Generating HDF5 files of training data and a text file
```
mkdir hdf5
python generate_training_data.py -1 /home/chpham/these/data/dHCP_ISBI2019/train/1_T1w_restore.nii.gz -o1 hdf5/1_T1w_restore.hdf5 -2 /home/chpham/these/data/dHCP_ISBI2019/train/1_T2w_restore.nii.gz -o2 hdf5/1_T2w_restore.hdf5 

```
1 : reference T1w image
o1 : HDF5 file which contains the patches of reference T1w image

2 : reference T2w image
o2 : HDF5 file which contains the patches of reference T2w image

Other arguments see : 
```
python generate_training_data.py 
```
##### Step 2 : Training networks
```
python UnSynGAN_train
```
Other arguments see : 
```
python UnSynGAN_train -h
```

