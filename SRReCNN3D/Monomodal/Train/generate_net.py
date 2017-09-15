# -*- coding: utf-8 -*-
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

import sys

sys.path.insert(0, './model')
from SRReCNN3D_net import SRReCNN3D_net, SRReCNN3D_deploy

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('-b','--batch', help='Indicates batch size for HDF5 storage', type=int, default=64)
    parser.add_argument('-l','--layers', help='Indicates number of layers of network (default=10)', type=int, default=10)
    parser.add_argument('-k','--kernel', help='Indicates size of filter (default=3)', type=int, default=3)
    parser.add_argument('--numkernel', help='Indicates number of filters (default=64)', type=int, default=64)
    parser.add_argument('-r','--residual', help='Using residual learning or None (default=True)', type=str, default='True')
    parser.add_argument('-t', '--text', help='Name of a text (.txt) file which contains HDF5 file names (default: model/train.txt)', type=str, default='model/train.txt')
    parser.add_argument('-n', '--netname', help='Name of train netwotk protocol (default=model/SRReCNN3D_net.prototxt)', type=str, default='model/SRReCNN3D_net.prototxt')
    parser.add_argument('-d', '--deployname', help='Name of deploy files in order to deploy the parameters of SRReCNN3D_net without reading HDF5 files (default=model/SRReCNN3D_deploy.prototxt)', type=str, default='model/SRReCNN3D_deploy.prototxt')
    
    args = parser.parse_args()
    
    #  ==== Parser  ===
    padding = int((args.kernel - 1)/float(2))
      
    # Check residual learning mode
    if args.residual == 'True':
        residual = True
    elif args.residual == 'False':
        residual = False
    else:
        raise AssertionError, 'Not support this residual mode. Try True or False !' 
    
    # Writing a text (.txt) file which contains HDF5 file names 
    OutFile = open(str(args.text), "w")
    
    # =========== Wrinting net ==================  
    with open(args.netname , 'w') as f:
        f.write(str(SRReCNN3D_net(args.text, args.batch, args.layers, args.kernel, args.numkernel, padding, residual)))
    SRReCNN3D_deploy(args.netname, args.deployname)
