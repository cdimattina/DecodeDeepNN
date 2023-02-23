#==========================================================
# File   :   localResponseNormaliation.py
# Author :   J.Burnham
# Date   :   04/5/2022
# Purpose:   Implement local response normalization as a keras layer
#==========================================================

import tensorflow as tf
from tensorflow import keras

"""Keras layer implementation of LRN
"""
class LocalResponseNormalization(keras.layers.Layer):

    def __init__(self, depth_radius, bias, alpha, beta, name=None):
        """Default Constructor of the LocalResponseNormalization class

        Args:
            depth_radius (int): defaults to 5. 0-D. Half-width of the 1-D normalization window
            bias (float): Defaults to 1. An offset (usally positive to avoid div by 0)
            alpha (float): Defaults to 1. A scale factor, usually positve
            beta (float): Dfaults to 0.5. An exponent
            name (float, optional): the name of the layer passed to Keras. Defaults to None.
        """
        super(LocalResponseNormalization, self).__init__(name=name)
        
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
    
    @tf.function
    def call(self, x):
        """Perform LRN

        Args:
            x (TF Tensor): the data passed into the layer

        Returns:
            TF Tensor: The data with LRN performed on it
        """
        x = tf.nn.local_response_normalization(x, self.depth_radius,
                                               self.bias, self.alpha, self.beta)

        return x