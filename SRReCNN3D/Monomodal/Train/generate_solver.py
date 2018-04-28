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
import argparse

string_init = """ # The train/test net protocol buffer definition
net: "model/SRReCNN3D_net.prototxt"
test_initialization: false   # We do not have testing phase during training
base_lr: 0.0001
momentum: 0.9
momentum2: 0.999
weight_decay: 0
lr_policy: "multistep"   # Decreasing learning rate after a stepvalue by multiplying with gamma
gamma: 0.1
stepvalue: 10000
max_iter: 20000      # Iteration/epoch = (number of patches)/batch.
display: 10
snapshot: 100
snapshot_prefix: "caffe_model/SRReCNN3D"
solver_mode: GPU
type: "Adam"
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net', help='The text file of network definition. (default=model/SRReCNN3D_net.prototxt)', type=str, default='model/SRReCNN3D_net.prototxt')
    parser.add_argument('-s', '--solver', help='Solver : optimization method (default=Adam)', type=str, default='Adam')
    parser.add_argument('-l', '--learningrate', help='Learning rate (default=0.001)', type=str, default='0.001')
    parser.add_argument('--momentum', help='Momentum (default=0.9)', type=str, default='0.9')
    parser.add_argument('--momentum2', help='Momentum2 (default=0.999)', type=str, default='0.999')
    parser.add_argument('-w', '--weightdecay', help='Weight decay (default=0)', type=str, default='0')
    parser.add_argument('--snapshot', help='Weight decay (default=100)', type=str, default='100')
    parser.add_argument('--stepvalue', help='Step value for decreasing learning rate', type=str, default='10000')
    parser.add_argument('-m', '--maxiter', help='Max iteration: Iteration/epoch = (number of patches)/batch.', type=str, default='20000')
    parser.add_argument('-o', '--outpath', help='Snapshot Prefix', type=str, default='caffe_model/SRReCNN3D')
    parser.add_argument('-t', '--text', help='The solver definition protocol buffer text file. (default=model/SRReCNN3D_solver.prototxt)', type=str, default='model/SRReCNN3D_solver.prototxt')
    args = parser.parse_args()       

    string = string_init
    string = string.replace('model/SRReCNN3D_net.prototxt',args.net)
    string = string.replace('type: "Adam"','type: "'+args.solver+'"')
    string = string.replace('base_lr: 0.0001','base_lr: '+args.learningrate)
    string = string.replace('momentum: 0.9','momentum: ' + args.momentum)
    string = string.replace('momentum2: 0.999','momentum2: ' + args.momentum2)
    string = string.replace('weight_decay: 0','weight_decay: ' + args.weightdecay)
    string = string.replace('snapshot: 100','snapshot: ' + args.snapshot)
    string = string.replace('stepvalue: 10000','stepvalue: ' + args.stepvalue)
    string = string.replace('max_iter: 20000','max_iter: ' + args.maxiter)
    string = string.replace('snapshot_prefix: "caffe_model/SRReCNN3D"','snapshot_prefix: "'+args.outpath+'"') 
        
    with open(args.text, "w") as f1:
        f1.write(string)
