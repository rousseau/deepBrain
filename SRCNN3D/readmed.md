# Brain MRI super-resolution using deep 3D convolutional networks

### Table of Contents

0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Dependencies](#dependencies)

### Introduction
This repository contains the SRCNN3D model described in the paper "Brain MRI super-resolution using deep 3D convolutional networks" (http://ieeexplore.ieee.org/abstract/document/7950500/).

### Citation

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
### Dependencies

#### For reading and writing NIFTI data:
[SimpleITK](https://itk.org/Wiki/SimpleITK/GettingStarted)

#### For training
[Caffe](https://github.com/BVLC/caffe/)

h5py
#### For GPU testing:
[Lasagne](https://lasagne.readthedocs.io/en/latest/)

[cudnn](https://developer.nvidia.com/cudnn)

### Testing

```
python demo_SRCNN3D.py -t ($Dataset)/KKI2009-01-MPRAGE_LR.nii.gz -r KKI2009-01-MPRAGE_LR_SRCNN3D.nii.gz -m caffe_model/SRCNN3D_iter_470000.caffemodel -n caffe_model/SRCNN3D_deploy.prototxt
```
t : testing LR image

r : SR image result

m : trained parameters of network stored in caffemodel file

n : deploy network

other arguments see : 
```
python demo_SRCNN3D.py -h
```

### Training
#### Step 1 : Generating HDF5 files of training data
```
python generate_hdf5.py -f ($Dataset)/KKI2009-33-MPRAGE.nii.gz -o hdf5/KKI2009-33-MPRAGE.hdf5 -f ($Dataset)/KKI2009-34-MPRAGE.nii.gz -o hdf5/KKI2009-34-MPRAGE.hdf5 -s 2,2,2 -s 3,3,3
```
f : HR reference image

o : HDF5 file which contains the patches of HR reference image

s : scale factor

other arguments see : 
```
python generate_hdf5.py -h
```
