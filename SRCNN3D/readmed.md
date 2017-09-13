# Brain MRI super-resolution using deep 3D convolutional networks

### Table of Contents

0. [Introduction](#introduction)
0. [Citation](#citation)

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
### Testing

```
python demo_SRCNN3D.py -t ($Dataset)/KKI2009-01-MPRAGE_LR.nii.gz -r KKI2009-01-MPRAGE_LR_SRCNN3D.nii.gz -m caffe_model/SRCNN3D_iter_470000.caffemodel -n caffe_model/SRCNN3D_deploy.prototxt
```
