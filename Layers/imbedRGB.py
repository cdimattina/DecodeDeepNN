#==========================================================
# File   :   imbedRGB.py
# Author :   J.Burnham
# Date   :   02/21/2022
# Purpose:   Imbed an greyscale image dataset into a larger rbg one
#==========================================================

import tensorflow as tf
from tensorflow import keras


"""Class soley for imbedding an image into a larger image, the smaller image would be surrounded by zeros.
   And converting that image into a RGB image
"""
class ImbedRGB(keras.layers.Layer):

    def __init__(self, desired_width, desired_height, name=None):
        """Default Constructor for ImbedRGB

        Args:
            desired_width (int)    : the width of the image to imbed the smaller into
            desired_height (int)   : the height of the image to imbed the smaller into
            name (string, optional): the name of the layer to pass to Keras. Defaults to None.
        """
        super(ImbedRGB, self).__init__(name=name)
        self.desired_width = desired_width
        self.desired_height = desired_height

    @tf.function

    def call(self, x):
        """_summary_

        Args:
            x (TF Tensor):  the data being pushed into the layer

        Returns:
            TF Tensor    :  the original data imbedded into an image of desired width and height
                            and converted to rgb
        """


        if(x.shape[3] ==1):
            x = tf.concat([x,x,x], axis=3) # convert to rgb

        if(x.shape[1] < 227):
            x = tf.image.pad_to_bounding_box(x,self.desired_height//2 - 20,self.desired_width//2 - 20,
                                            self.desired_height,self.desired_width)

        return x
