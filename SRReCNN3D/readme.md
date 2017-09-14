# In progress


### Table of Contents
1. [Introduction](#introduction)
1. [Dependencies](#dependencies)
1. [Testing](#testing)

### 1. Introduction
This repository contains the SRReCNN3D model for brain MRI super-resolution

### 2. Dependencies

#### For reading and writing NIFTI data:
[SimpleITK](https://itk.org/Wiki/SimpleITK/GettingStarted)

#### For training
[Caffe](https://github.com/BVLC/caffe/)

h5py

#### For testing:
[Lasagne](https://lasagne.readthedocs.io/en/latest/)

[cudnn](https://developer.nvidia.com/cudnn)

### 3. Monomodal

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


