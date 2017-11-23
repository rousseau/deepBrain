# SRReCNN3D (In progress)


### Table of Contents
0. [Introduction](#introduction)
1. [Citation](#citation)
1. [Dependencies](#dependencies)
1. [Monomodal brain MRI SR](#monomodal)
1. [Multimodal brain MRI SR](#multimodal)

### 0. Introduction
This repository contains the SRReCNN3D model for brain MRI super-resolution

### 1. Citation

If you use these models in your research, please cite:
```
@article{pham2017multi,
  title={Multi-scale brain MRI super-resolution using deep 3D convolutional networks},
  author={Pham, Chi-Hieu and Fablet, Ronan and Rousseau, Fran{\c{c}}ois},
  journal={HAL preprint hal-01635455},
  year={2017}
}
```

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

#### b) Training
##### Step 1 : Generating HDF5 files of training data and a text file of network protocol
```
cd Monomodal/Train
python generate_training.py -f ($Dataset)/KKI2009-33-MPRAGE.nii.gz -o hdf5/KKI2009-33-MPRAGE.hdf5 -f ($Dataset)/KKI2009-34-MPRAGE.nii.gz -o hdf5/KKI2009-34-MPRAGE.hdf5 -s 2,2,2 -s 3,3,3 -l 10 -k 3 --numkernel 32
```
f : HR reference image

o : HDF5 file which contains the patches of HR reference image

s : scale factors

l : number of layers

k: size of 3d filter ![](https://latex.codecogs.com/gif.latex?k%20%5Ctimes%20k%20%5Ctimes%20k)

numkernel : number of filters

Other arguments see : 
```
python generate_training.py -h
```

**Optional 1 :** After having HDF5 files, if we would like to modify network but do not want to generate them again.

```
python generate_net.py -l 5 -k 3 --numkernel 64
```

**Optional 2 :** If we have LR images and corresponding HR images and we do not want to use the following observation model:


![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BY%7D%20%3D%20%5CTheta%20%5Ctextbf%7BX%7D%20&plus;%20%5Ctextbf%7BN%7D%20%3D%20D_%7B%5Cdownarrow%7D%20B%5Ctextbf%7BX%7D%20&plus;%20%5Ctextbf%7BN%7D)

where :

Y : observed LR image

X : HR image

![](https://latex.codecogs.com/gif.latex?D_%7B%5Cdownarrow%7D) : downscaling operator

B : blurring filter

We can use this function to generating HDF5 files and network protocol. Of course, this function can be used for other tasks using CNNs.

```
python generate_training_free.py -f ($Dataset)/KKI2009-33-MPRAGE.nii.gz -i ($Dataset)/KKI2009-33-MPRAGE_LR.nii.gz -o hdf5/KKI2009-33-MPRAGE.hdf5 

```

f : label HR image

i : input LR image

o : HDF5 file which contains the patches of HR reference image

Other arguments see : 

```
python generate_training_free.py -h
```

#### Step 2 : Creating a solver text file for Caffe
We can modify the solver text file at *model/SRReCNN3D_solver.prototxt*

For further information about Caffe solver at [here](http://caffe.berkeleyvision.org/tutorial/solver.html)

**Or** using this function:
```
python generate_solver.py -l 0.001 -s Adam
```
l : learning rate

s : optimization method

#### Step 3 : Training network using Caffe
```
mkdir caffe_model
caffe train --solver model/SRReCNN3D_solver.prototxt
```


### 4. Multimodal

#### a) Testing

Testing multimodal 10-layers residual network (Multimodal 10L-ReCNN) with isotropic scale factor of 2 for brain MRI super-resolution and with the help of a (registered or not) HR reference image.

```
cd Multimodal/Test
python demo_multiSRReCNN3D.py -t ($Dataset)/KKI2009-01-MPRAGE_LR.nii.gz -f ($Dataset)/KKI2009-01-T2w.nii.gz -r KKI2009-01-MPRAGE_LR_MonoSRReCNN3D.nii.gz -m caffe_model/MultiSRReCNN3D_10Layers_IsoScalex2.caffemodel -n caffe_model/MultiSRReCNN3D_10L_deploy.prototxt
```
t : testing LR image

f : HR reference image

r : SR image result

m : trained parameters of network stored in caffemodel file

n : deploy network

Other arguments see : 
```
python demo_multiSRReCNN3D.py -h
```

#### b) Training

##### Step 1 : Generating HDF5 files of training data and a text file of network protocol
```
cd Monomodal/Train
python generate_intermod_training.py -f ($Dataset)/KKI2009-33-MPRAGE.nii.gz -i ($Dataset)/KKI2009-33-T2w.nii.gz -o hdf5/KKI2009-33-MPRAGE.hdf5 -f ($Dataset)/KKI2009-34-MPRAGE.nii.gz -i ($Dataset)/KKI2009-34-T2w.nii.gz -o hdf5/KKI2009-34-MPRAGE.hdf5 -s 2,2,2 -s 3,3,3 -l 10 -k 3 --numkernel 32
```
f : HR reference image (ex T1-weighted MRI)

i : Intermodality HR reference image (ex T2-weighted MRI)

o : HDF5 file which contains the patches of HR reference image

s : scale factors

l : number of layers

k: size of 3d filter ![](https://latex.codecogs.com/gif.latex?k%20%5Ctimes%20k%20%5Ctimes%20k)

numkernel : number of filters

Other arguments see : 
```
python generate_intermod_training.py -h
```

**Optional :** After having HDF5 files, if we would like to modify network but do not want to generate them again.

```
python generate_intermod_net.py -l 5 -k 3 --numkernel 64
```


#### Step 2 : Creating a solver text file for Caffe
We can modify the solver text file at *model/SRReCNN3D_solver.prototxt*

For further information about Caffe solver at [here](http://caffe.berkeleyvision.org/tutorial/solver.html)

**Or** using this function:
```
python generate_solver.py -l 0.001 -s Adam
```
l : learning rate

s : optimization method

#### Step 3 : Training network using Caffe
```
mkdir caffe_model
caffe train --solver model/SRReCNN3D_solver.prototxt
```
