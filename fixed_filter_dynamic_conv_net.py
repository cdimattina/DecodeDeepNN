#==========================================================
# File      : dynamic_conv_net.py
# Author    : J.Burnham
# Purpose   : Implemenation File for the DynamicConvNet Class
#==========================================================

import numpy
from tensorflow import keras
from keras import models
from keras import layers

from layers.filter_bank import FilterBank

class DynamicConvNet:
    def __init__(self,hyper_parameters, outputs, img_size, num_channels,filters, pooling=True, pool_ksize=1, pool_strides=1):
        """constructor for DynamicConvNet class

        Args:
            hyper_parameters (numpy array): the hyperparameters and number of layers in the network
                                            stored in a numpy array of shape [number of layers, number of hyperparameters].
                                            The order of the hyper parameters are as follows
                                            [# of filters, width of filters, convolution strides, downsampling size, downsampling stride, l1 penatly, l2 penalty]

            outputs (int): the number of outputs the network should have
            :param filters:      - A 2D list of filters whose first dimension follows the pattern:
                                   [file path, filter name, number of filters]
            :param pooling:      - A Boolean for whether or Max Pooling is to be used
            :param pool_ksize:   - The kernel size for the Max Pooling Function
            :param pool_strides: - The strides for the Max Pooling Function
        """
        self.hyper_parameters = hyper_parameters
        self.outputs = outputs
        self.img_size = img_size
        self.num_channels = num_channels
        self.filters = filters
        self.pooling = pooling
        self.pool_ksize = pool_ksize
        self.pool_strides = pool_strides

    def generate_conv_net(self):
        """Dynamically generate a convolutional neural network by the 
           hyperparameters and number of outputs passed at instantiation.
           NOTE: model only consists of alternating Conv2D and Maxpooling layers
           with a flatten and output layer also added at the end.
        """
        model = models.Sequential() # create the empty model
        model.add(layers.Input(shape=(self.img_size,self.img_size,self.num_channels)))
        model.add(FilterBank(self.filters, self.pooling, self.pool_ksize, self.pool_strides ))

        for i in range(self.hyper_parameters.shape[0]):
            parameters = self.hyper_parameters[i]
            # [# of filters, width of filters, convolution strides, downsampling size, downsampling stride, l1 penatly, l2 penalty]
            model.add(layers.Conv2D(filters=parameters[0], kernel_size=(parameters[1], parameters[1]), strides=(parameters[2], parameters[2]),padding="same", name=f"Conv2D_Layer{i}"))
            model.add(layers.MaxPooling2D(pool_size=(parameters[3], parameters[3]), strides=(parameters[4], parameters[4]), name=f"MaxPooling2D_Layer{i}"))
        model.add(layers.Flatten(name="Output_Flatten"))
        model.add(layers.Dense(self.outputs, name="Output_Layer"))

        return model
