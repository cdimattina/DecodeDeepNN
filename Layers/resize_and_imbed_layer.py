# ==========================================================
# File      : dynamic_pretrained_model.py
# Author    : Josiah Burnham (2-23-2023)
# Purpose   : implementation of the DynamicPretrainedModel Class
# ==========================================================

import tensorflow as tf
from tensorflow import keras


class ResizeImbedRBG (keras.layers.Layer):

    def __init__(self, resize_width, resize_height, desired_width, desired_height, is_resize=True, is_imbed=True, is_rgb=True, name=None):
        super(ResizeImbedRBG, self).__init__(name=name)

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.is_resize = is_resize
        self.is_imbed = is_imbed
        self.is_rgb = is_imbed

    @tf.function
    def call(self, x):

        if self.is_resize:
            x = tf.image.resize(x, (self.resize_height, self.resize_width))

        if (x.shape[3] == 1 and self.is_rgb):
            x = tf.concat([x, x, x], axis=3)  # convert to rgb

        if (x.shape[1] < self.desired_height and self.is_imbed):
            x = tf.image.pad_to_bounding_box(x, self.desired_height//2 - self.resize_height//2, self.desired_width//2 - self.resize_width//2,
                                             self.desired_height, self.desired_width)

        return x
