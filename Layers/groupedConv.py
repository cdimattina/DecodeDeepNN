#==========================================================
# File   :   groupedConv.py
# Author :   J.Burnham
# Date   :   04/5/2022
# Purpose:   Implement a group convolution layer similar to the one used
#            in the original implementation of AlexNet
#==========================================================

import tensorflow as tf
from tensorflow import keras

""" two convolutional layers each passed half of the channels, then concatenated and returned
"""
class GroupedConv2D(keras.layers.Layer):

    def __init__(self, kernels, dim, name, trainable=True):
        """Default Constructor for GroupedConv2D class

        Args:
            kernels (int)              : number of kernels to use for both conv layers
            dim (tuple of ints)        : the dimensions of the kernels for both filters
            name (string)              : the name of the layer passed to keras
            trainable (bool, optional) : should the layers be trainable or not. Defaults to True.
        """
        super(GroupedConv2D, self).__init__(name=name)
        self.kernels = kernels
        self.dim = dim
        self.conv1 = keras.layers.Conv2D(self.kernels, self.dim, padding='same', activation='relu', name=name+"_1", trainable=trainable)
        self.conv2 = keras.layers.Conv2D(self.kernels,self.dim, padding='same', activation='relu', name=name+"_2", trainable=trainable)

    @tf.function
    def call(self, inputs):
        """pass the data through both convolutional layers

        Args:
            inputs (TF Tensor): the data passed into the layer

        Returns:
            TF Tensor: the data, cut in half, passed through each conv layer, concatenated back together
        """

        # split the channels in half to pass one to each filter
        num_channels = inputs.shape[-1]
        corrected_channels = num_channels//2
        inputs_1 = inputs[:,:,:, 0:corrected_channels]
        inputs_2 = inputs[:,:,:, corrected_channels:num_channels]
        

        inputs_1 = self.conv1(inputs_1)
        inputs_2 = self.conv2(inputs_2)

        # join the inputs back to one tensor
        output = tf.concat([inputs_1, inputs_2], axis=3)

        return output

    def setWeights(self,w1,b1,w2,b2, c1_shape, c2_shape): 
        """set the weights of each of the conv layers 

        Args:
            w1 (TF Tensor): Weight matrix for first conv layer
            b1 (TF Tensor): Bias for first conv layer
            w2 (TF Tensor): Weight matrix for second conv layer
            b2 (TF Tensor): Bias for second conv layer
            c1_shape (Tuple ints): input shape for the first layer
            c2_shape (Tuple ints): inputs shape for the second layer
        """
        self.conv1.build(c1_shape)
        self.conv2.build(c2_shape)

        self.conv1.set_weights([w1,b1])
        self.conv2.set_weights([w2,b2])