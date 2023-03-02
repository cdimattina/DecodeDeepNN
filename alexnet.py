# ==========================================================
# File   :   AlexNet.py
# Author :   J.Burnham
# Date   :   01/31/2022
# Purpose:  Implementation of the AlexNet Convolutional Neural Network into Keras
# ==========================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from Layers.groupedConv import GroupedConv2D
from Layers.imbedRGB import ImbedRGB
from Layers.localResponseNormalization import LocalResponseNormalization


class AlexNet():
    """ A implementation of the AlexNet Model, where the caller can specify what block
        of the model is desired.
    """

    def __init__(self, wd, num_outputs, l1, inputs, pool_rate, output=True, prog_pool=True):
        """property constructor for AlexNet class

        Args:
            wd (array[numpy arrays]): weight array of numpy arrays
            num_outputs (int): number of perdiction classes
        """

        self.wd = wd
        self.num_outputs = num_outputs
        self.output = output
        self.l1 = l1
        self.pool_rate = pool_rate
        self.prog_pool = prog_pool
        self.inputs = inputs

        # Conv Layers
        # ----------------------------------------
        self.conv1 = layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(
            4, 4), padding='same', activation="relu", name="Conv1", trainable=False)
        self.conv2 = GroupedConv2D(
            128, (5, 5), name="GroupConv1", trainable=False)
        self.conv3 = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(
            1, 1), padding='same', activation="relu", name="Conv2", trainable=False)
        self.conv4 = GroupedConv2D(
            192, (3, 3), name="GroupConv2", trainable=False)
        self.conv5 = GroupedConv2D(
            128, (3, 3), name="GroupConv3", trainable=False)

        # Fully Connected Layers
        # ----------------------------------------
        self.hidden_dense1 = layers.Dense(
            4096, activation="relu", name="Hidden1", trainable=False)
        self.hidden_dense2 = layers.Dense(
            4096, activation="relu", name="Hidden2", trainable=False)
        # self.output_layer = layers.Dense(num_outputs,  activation="softmax", name="Output", trainable=True,  kernel_regularizer=keras.regularizers.l2(self.l2))
        self.output_layer = layers.Dense(
            num_outputs, name="Output", trainable=True, kernel_regularizer=keras.regularizers.l1(self.l1))

        # Intermediary Layers
        # ----------------------------------------
        self.flatten = layers.Flatten()
        self.imbedLayer = ImbedRGB(227, 227)  # 227x227 is what AlexNet expects

    def get_model(self, layer_level):
        """get AlexNet varient with only specified blocks

        Args:
            layer_level (int): the block layer that is desired (inclusive)

        Returns:
            keras.model: a AlexNet model that only contains the specified blocks
        """
        ds_size = 1  # so multiplying doesn't give me a 0
        self.__set_layer_weights()

        model = keras.Sequential([], name="Alexnet")

        if (layer_level >= 1):  # block 1
            ds_size = self.pool_rate
            model.add(layers.Input(self.inputs))
            model.add(self.imbedLayer)
            model.add(self.conv1)
        if (layer_level >= 2):
            model.add(LocalResponseNormalization(depth_radius=2,
                      bias=5, alpha=0.0001, beta=0.75, name="LRN1"))
        if (layer_level >= 3 and not self.prog_pool):
            model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(
                2, 2), padding='valid', data_format="channels_last", name="MaxPool_1"))
        if (layer_level == 3 and self.prog_pool):
            model.add(layers.MaxPooling2D(pool_size=(ds_size, ds_size), strides=(
                ds_size, ds_size), padding='valid', data_format="channels_last", name="prog_pool"))

        if (layer_level >= 4):  # block 2
            ds_size = self.pool_rate
            model.add(self.conv2)
        if (layer_level >= 5):
            model.add(LocalResponseNormalization(depth_radius=2,
                      bias=5, alpha=0.0001, beta=0.75, name="LRN2"))
        if (layer_level >= 6 and not self.prog_pool):
            model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(
                2, 2), padding='valid', data_format="channels_last", name="MaxPool_2"))
        if (layer_level == 6 and self.prog_pool):
            model.add(layers.MaxPooling2D(pool_size=(ds_size, ds_size), strides=(
                ds_size, ds_size), padding='valid', data_format="channels_last", name="prog_pool"))

        if (layer_level >= 7):  # block 3
            ds_size = self.pool_rate
            model.add(self.conv3)
        if (layer_level >= 8):
            model.add(LocalResponseNormalization(depth_radius=2,
                      bias=5, alpha=0.0001, beta=0.75, name="LRN3"))
        if (layer_level >= 9):
            model.add(self.conv4)
        if (layer_level >= 10):
            model.add(LocalResponseNormalization(depth_radius=2,
                      bias=5, alpha=0.0001, beta=0.75, name="LRN4"))
        if (layer_level >= 11):
            model.add(self.conv5)
        if (layer_level >= 12):
            model.add(LocalResponseNormalization(depth_radius=2,
                      bias=5, alpha=0.0001, beta=0.75, name="LRN5"))
        if (layer_level >= 13 and not self.prog_pool):
            model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(
                2, 2), padding='valid', data_format="channels_last", name="MaxPool_3"))
        if (layer_level == 13 and self.prog_pool):
            model.add(layers.MaxPooling2D(pool_size=(ds_size, ds_size), strides=(
                ds_size, ds_size), padding='valid', data_format="channels_last", name="prog_pool"))

        if (layer_level >= 14):  # block 4
            ds_size = self.pool_rate
            model.add(self.flatten)
        if (layer_level >= 15):
            model.add(self.hidden_dense1)
        if (layer_level >= 16):
            model.add(layers.Dropout(0.0))
        if (layer_level >= 17):
            model.add(self.hidden_dense2)
        if (layer_level >= 18):
            model.add(layers.Dropout(0.0))

        if (self.prog_pool and layer_level not in [3, 6, 13]):
            model.add(layers.MaxPooling2D(pool_size=(ds_size, ds_size), strides=(
                ds_size, ds_size), padding='valid', data_format="channels_last", name="prog_pool"))

        if (layer_level < 14 and self.output):
            model.add(self.flatten)

        if (self.output):
            model.add(layers.Dense(self.outputs,trainable=True,kernel_regularizer = keras.regularizers.l1(self.l1)))

        return model

    def __set_layer_weights(self):
        """set the weights of all the layers in the model, from the weigths that were passed
            at instantiation
        """

        # Convolution Layers
        # ----------------------------------------
        # Build The Layers
        self.conv1.build((227, 227, 3))
        self.conv3.build((1, 27, 27, 256))

        # Set the weights
        self.conv1.set_weights([self.wd[0], tf.squeeze(self.wd[1])])
        self.conv2.setWeights(self.wd[2], tf.squeeze(self.wd[3]), self.wd[4], tf.squeeze(
            self.wd[5]), (1, 55, 55, 48), (1, 55, 55, 48))
        self.conv3.set_weights([self.wd[6], tf.squeeze(self.wd[7])])
        self.conv4.setWeights(self.wd[8], tf.squeeze(self.wd[9]), self.wd[10], tf.squeeze(
            self.wd[11]), (1, 27, 27, 192), (1, 27, 27, 192))
        self.conv5.setWeights(self.wd[12], tf.squeeze(self.wd[13]), self.wd[14], tf.squeeze(
            self.wd[15]), (1, 27, 27, 192), (1, 27, 27, 192))

        # Fully Connected Layers
        # ---------------------------------------
        # Build The Layers
        self.hidden_dense1.build((1, 9216))
        self.hidden_dense2.build((1, 4096))

        # Set the weights
        # matlab reverses the dims for there FC layer
        self.wd[16] = tf.reshape(self.wd[16], (9216, 4096))
        self.hidden_dense1.set_weights([self.wd[16], tf.squeeze(self.wd[17])])
        self.hidden_dense2.set_weights([self.wd[18], tf.squeeze(self.wd[19])])
