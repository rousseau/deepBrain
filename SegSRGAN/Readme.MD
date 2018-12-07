# SegSRGAN (In progress)


### Table of Contents
0. [Introduction](#introduction)
1. [Citation](#citation)
1. [Dependencies](#dependencies)
1. [20L-SRReCNN3D](#20L-SRReCNN3D)
1. [SegSRGAN](#SegSRGAN)

### 0. Introduction
This repository contains the SRReCNN3D model for brain MRI super-resolution

### 1. Citation

Submitted to ISBI2019

### 2. Dependencies

#### For reading and writing NIFTI data:
[SimpleITK](https://itk.org/Wiki/SimpleITK/GettingStarted)

#### For training
[Keras](https://keras.io/)

h5py

### 3. 20L-SRReCNN3D

#### a) Testing

Testing monomodal 10-layers residual network (Monomodal 10L-ReCNN) with isotropic scale factor of 2 for brain MRI super-resolution.

```
cd Monomodal/Test
python demo_monoSRReCNN3D.py -t ($Dataset)/KKI2009-01-MPRAGE_LR.nii.gz -r KKI2009-01-MPRAGE_LR_MonoSRReCNN3D.nii.gz -m caffe_model/SRReCNN3D_10Layers_IsoScalex2.caffemodel -n caffe_model/SRReCNN3D_10L_deploy.prototxt -l 10 -s 2,2,2
```
Testing multiscale monomodal 20-layers residual network (Multiscale Monomodal 20L-ReCNN) with isotropic scale factor of 2 and 3 for brain MRI super-resolution.

```
python demo_monoSRReCNN3D.py -t ($Dataset)/KKI2009-01-MPRAGE_LR.nii.gz -r KKI2009-01-MPRAGE_LR_MonoSRReCNN3D.nii.gz -m caffe_model/SRReCNN3D_20Layers_IsoScalex2x3.caffemodel -n caffe_model/SRReCNN3D_20L_deploy.prototxt -l 20 -s 3,3,3

```
t : testing LR image

r : SR image result

m : trained parameters of network stored in caffemodel file

n : deploy network

l : number of layers of network

s : scale factor (ex 3,3,3 or 2.5,2.5,2.5)

Other arguments see : 
```
python demo_monoSRReCNN3D.py -h
```

#### b) Training
##### Step 1 : Generating HDF5 files of training data and a text file
```
mkdir hdf5
python SRReCNN3D_generate_training_data.py -f ($Dataset)/1_T2w_restore.nii.gz -o hdf5/1_T2w_restore.hdf5 -f ($Dataset)/2_T2w_restore.nii.gz -o hdf5/2_T2w_restore.hdf5 -n 
```
f : HR reference image

o : HDF5 file which contains the patches of HR reference image

n : new low resolution (ex: 0.46x0.46x6 mm)

Other arguments see : 
```
python SRReCNN3D_generate_training_data.py -h
```
#### Step 2 : Training networks
```
python SRReCNN3D_train.py
```
Other arguments see : 
```
python SRReCNN3D_train.py -h


```
