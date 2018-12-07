# SegSRGAN (In progress)


### Table of Contents
0. [Introduction](#introduction)
1. [Citation](#citation)
1. [Dependencies](#dependencies)
1. [20L-SRReCNN3D](#20L-SRReCNN3D)
1. [SegSRGAN](#SegSRGAN)

### 0. Introduction
This repository contains the SRReCNN3D network for brain MRI super-resolution and the SegSRGAN network for simultaneous super-resolution and segmentation using a generative adversarial network. These models can be applied to clinical low-resolution data.

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

Testing monomodal 20-layers residual network (20L-SRReCNN3D) for low-resolution image with the resolution of 0.4464x0.4464x3 mm in order to generate isotropic SR image of 0.5x0.5x0.5 mm

```
python SRReCNN3D_test.py -t ($Dataset)/S00096_T2_TE200_corrected.nii.gz -o S00096_T2_TE200_corrected_SR.nii.gz -w weights/SRReCNN3D
```
t : testing low-resolution image

o : output SR result

Other arguments see : 
```
python SRReCNN3D_test.py -h
```

#### b) Training
##### Step 1 : Generating HDF5 files of training data and a text file
```
mkdir hdf5
python SRReCNN3D_generate_training_data.py -f ($Dataset)/1_T2w_restore.nii.gz -o hdf5/1_T2w_restore.hdf5 -f ($Dataset)/2_T2w_restore.nii.gz -o hdf5/2_T2w_restore.hdf5 -n 0.4464,0.4464,3
```
f : HR reference image

o : HDF5 file which contains the patches of HR reference image

n : simulated low resolution (ex: 0.4464x0.4464x3 mm)

Other arguments see : 
```
python SRReCNN3D_generate_training_data.py -h
```
##### Step 2 : Training networks
```
python SRReCNN3D_train.py
```
Other arguments see : 
```
python SRReCNN3D_train.py -h
```

### 4. SegSRGAN

#### a) Testing

Testing monomodal 20-layers residual network (20L-SRReCNN3D) for low-resolution image with the resolution of 0.4464x0.4464x3 mm in order to generate isotropic SR image of 0.5x0.5x0.5 mm

```
python SegSRGAN_test.py -t ($Dataset)/S00096_T2_TE200_corrected.nii.gz -o S00096_T2_TE200_corrected_SR.nii.gz -c S00096_T2_TE200_corrected_cortex.nii.gz -w weights/SegSRGAN
```
t : testing low-resolution image

o : output SR result

c : output cortex segmentation result

Other arguments see : 
```
python SegSRGAN_test.py
```

#### b) Training
##### Step 1 : Generating HDF5 files of training data and a text file
```
mkdir hdf5
python SegSRGAN_generate_training_data.py -f ($Dataset)/1_T2w_restore.nii.gz -c ($Dataset)/1_ribbon_cortex.nii.gz -o hdf5/1_T2w_restore.hdf5 -f ($Dataset)/2_T2w_restore.nii.gz -c ($Dataset)/2_ribbon_cortex.nii.gz -o hdf5/2_T2w_restore.hdf5 -n 0.4464,0.4464,3
```
f : HR reference image

c : HR reference cortex segmentation

o : HDF5 file which contains the patches of HR reference image

n : simulated low resolution (ex: 0.4464x0.4464x3 mm)

Other arguments see : 
```
python SegSRGAN_generate_training_data.py -h
```
##### Step 2 : Training networks
```
python SegSRGAN_train.py
```
Other arguments see : 
```
python SegSRGAN_train.py -h
```
