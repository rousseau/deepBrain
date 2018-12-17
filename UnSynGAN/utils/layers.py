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

from tensorflow.python.ops import array_ops
from keras.engine.topology import Layer
import keras.backend as K
import numpy as np
from keras.layers import Multiply

def charbonnier_loss(y_true, y_pred):
    """
    https://en.wikipedia.org/wiki/Huber_loss
    """
    epsilon = 1e-3
    diff = y_true - y_pred
    loss = K.mean(K.sqrt(K.square(diff)+epsilon*epsilon), axis=-1)
    return K.mean(loss)

class ConditionalInstanceNormalization3D(Layer):
    ''' https://arxiv.org/pdf/1703.06868.pdf'''
    def __init__(self, **kwargs):
        super(ConditionalInstanceNormalization3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[0][1],), initializer="one", trainable=True)
        self.shift = self.add_weight(name='shift', shape=(input_shape[0][1],), initializer="zero", trainable=True)
        super(ConditionalInstanceNormalization3D, self).build(input_shape)

    def call(self, inputs):
        def image_expand(tensor):
            return K.expand_dims(K.expand_dims(K.expand_dims(tensor, -1), -1), -1)

        def batch_image_expand(tensor):
            return image_expand(K.expand_dims(tensor, 0))

        x = inputs[0]
        hwk = K.cast(x.shape[2] * x.shape[3] * x.shape[4], K.floatx())
        mu = K.sum(x, [-1, -2, -3]) / hwk
        mu_vec = image_expand(mu) 
        sig2 = K.sum(K.square(x - mu_vec), [-1, -2, -3]) / hwk
        y = (x - mu_vec) / (K.sqrt(image_expand(sig2)) + K.epsilon())

        embed_weight = inputs[1]
        scale = batch_image_expand(self.scale)
        shift = batch_image_expand(self.shift)
        scale = Multiply()([scale ,embed_weight])
        shift = Multiply()([shift ,embed_weight])
        return scale*y + shift 

    def compute_output_shape(self, input_shape):
        return input_shape[0] 

def total_variation_loss_3D(y_true, y_pred, beta=2.):
    """
    https://arxiv.org/pdf/1412.0035.pdf
    """
    W, H, D = K.shape(y_pred)[2], K.shape(y_pred)[3], K.shape(y_pred)[4]
    pixel_dif1 = K.square(y_pred[:, :, :W-1,:H-1,:D-1] - y_pred[:, :, 1:,:H-1,:D-1])
    pixel_dif2 = K.square(y_pred[:, :, :W-1,:H-1,:D-1] - y_pred[:, :, :W-1,1:,:D-1])
    pixel_dif3 = K.square(y_pred[:, :, :W-1,:H-1,:D-1] - y_pred[:, :, :W-1,:H-1,1:])
    pixel_dif = pixel_dif1 + pixel_dif2 + pixel_dif3
    return K.sum(K.pow(pixel_dif, beta/2.))

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)

class ReflectPadding3D(Layer):
    def __init__(self, padding=1, **kwargs):
        super(ReflectPadding3D, self).__init__(**kwargs)
        self.padding = ((padding, padding), (padding, padding), (padding, padding))

    def compute_output_shape(self, input_shape):
        if input_shape[2] is not None:
            dim1 = input_shape[2] + self.padding[0][0] + self.padding[0][1]
        else:
            dim1 = None
        if input_shape[3] is not None:
            dim2 = input_shape[3] + self.padding[1][0] + self.padding[1][1]
        else:
            dim2 = None
        if input_shape[4] is not None:
            dim3 = input_shape[4] + self.padding[2][0] + self.padding[2][1]
        else:
            dim3 = None
        return (input_shape[0],
                input_shape[1],
                dim1,
                dim2,
                dim3)

    def call(self, inputs):
        pattern = [[0, 0], [0, 0], 
                   [self.padding[0][0], self.padding[0][1]],
                   [self.padding[1][0], self.padding[1][1]], 
                   [self.padding[2][0], self.padding[2][1]]]
            
        return array_ops.pad(inputs, pattern, mode= "REFLECT")

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
