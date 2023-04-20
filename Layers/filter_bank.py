"""
File:   filter_bank.py
Author: Josiah Burnham (8/2021)
Org:    FGCU
Desc:   Custom Keras Layer that is a Fixed Filter Bank
"""
import keras.layers
import numpy as np
import tensorflow as tf
import scipy.io


class FilterBank(keras.layers.Layer):
    """
    A Custom Keras Layer, that is deigned to convolve inputted images, with inputted fixed filters in .mat files
    """

    def __init__(self, filters, pooling=True, pool_ksize=1, pool_strides=1):
        """
        Standard Constructor for FilerBank Class

        :param filters:      - A 2D list of filters whose first dimension follows the pattern:
                               [file path, filter name, number of filters]
        :param pooling:      - A Boolean for whether or Max Pooling is to be used
        :param pool_ksize:   - The kernel size for the Max Pooling Function
        :param pool_strides: - The strides for the Max Pooling Function
        """
        super(FilterBank, self).__init__()
        self.filters = filters
        self.filter_bank = self._load_filters()

        self.pooling = pooling
        self.pool_ksize = pool_ksize
        self.pool_strides = pool_strides

    def call(self, x):
        """
        The Function that is called when the Instantiated class object is called

        :param x: - The input whether it be directly or handled by keras itself
        :return:  - A 4D Tensor with shape (Batch Size, Image Width, Image Height, Number of Filtered Copies)
        """

        x = self._filter_images(x)
        if self.pooling:
            x = self._max_pool(x)

        return x

    def _filter_images(self, input_image):
        """
        Takes a Image which is a 4D tensor with shape (Batch Size, Image Width, Image Height, Number of Color Channels),
         and convolve it with a Filter that is also a 4D Tensor with shape:
        (Filter Height, Filter Width, Input Channels, Output Channels).

        It will then append it to a Numpy Array, and move on to the next filter if applicable

        :param input_image: - A 4D Tensor with Shape: (Batch Size, Image Width, Image Height, Number of Color Channels)
        :return: numpy array - A 4D Tensor with Shape (Batch Size, Image Width, Image Height, Number of Filtered Copies)
        """

        filtered_images = np.ndarray(shape=(input_image.shape[0], input_image.shape[1], input_image.shape[2], 0),
                                     dtype=float)

        for i in range(len(self.filter_bank)):
            cur_filter = self.filter_bank[i]
            cur_filter = tf.Variable(
                cur_filter, dtype=tf.float32, trainable=False)

            filtered_image = self._convolution(input_image, cur_filter)

            filtered_images = np.append(
                filtered_images, filtered_image, axis=3)

        return filtered_images

    def _load_filters(self):
        """
        Loads the passed filters from a .mat file

        :return: array- returns the loaded filters in an array
        """

        filter_bank = []
        for i in range(len(self.filters)):
            loaded_filters = scipy.io.loadmat(self.filters[i][0])[
                self.filters[i][1]]
            filter_bank.append(loaded_filters)

        return filter_bank

    @staticmethod
    def _convolution(input_image, cur_filter):
        """
        Convolve a 4D image tensor with a 4D filter Tensor

        :param input_image: - The 4D image tensor with shape:
                              (Batch Size, Image Width, Image Height, Number of Color Channels)

        :param cur_filter:  - The 4D filter tensor of shape:
                              (Filter Height, Filter Width, Input Channels, Output Channels) to convolve with the image

        :return: the relu of the convolution calculation
        """
        input_data = tf.nn.conv2d(input_image, cur_filter, strides=[
                                  1, 1, 1, 1], padding="SAME")
        return tf.nn.relu(input_data)

    def _max_pool(self, input_img):
        """
        Preforms Max Pooling on the inputted 4D image Tensor

        :param input_img: - The 4D image Tensor with shape:
                            (Batch Size, Image Width, Image Height, Number of Color Channels)

        :return: The Max pooling calculation
        """
        return tf.nn.max_pool(input_img, ksize=[1, self.pool_ksize, self.pool_ksize, 1],
                              strides=[1, self.pool_strides, self.pool_strides, 1], padding="SAME")
