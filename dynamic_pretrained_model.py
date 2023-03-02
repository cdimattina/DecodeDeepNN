# ==========================================================
# File      : dynamic_pretrained_model.py
# Author    : Josiah Burnham (2-23-2023)
# Purpose   : implementation of the DynamicPretrainedModel Class
# ==========================================================

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications import *
from Layers.imbedRGB import ImbedRGB
import numpy as np

from alexnet import AlexNet
from Layers.resize_and_imbed_layer import ResizeImbedRBG

class DynamicPretrainedModel:
    def __init__(self, model_name, layer_number, l1, inputs, outputs):
        """Constructor for the DynamicPretrainedModel Class

        Args:
            model_name (string): the name of the pretrained model
            layer_number (int): the number of layer to return a model of
            inputs (int tuple): the input shape for the returned model
        """
        self.model_name = model_name
        self.layer_number = layer_number
        self.inputs = inputs
        self.outputs = outputs
        self.l1      = l1

        self.imbedLayer = ResizeImbedRBG(
            64, 64, 224, 224, is_resize=False, is_imbed=False, is_rgb=True)  # 227x227 is what AlexNet expects

    def __generate_model(self):
        """ download the pretrained model to get the individual layer
            activation of

        Returns:
            keras.Model: the pretrain model to get the layers of
        """
        if self.model_name == "VGG16":
            model = vgg16.VGG16(weights='imagenet', include_top=False)
            model.trainable = False

        if self.model_name == "VGG19":
            model = vgg19.VGG19(weights='imagenet', include_top=False)
            model.trainable = False

        if self.model_name == "ResNet50":
            model = resnet50.ResNet50(weights='imagenet', include_top=False)
            model.trainable = False

        if self.model_name == "InceptionV3":
            model = inception_v3.InceptionV3(
                weights='imagenet', include_top=False)
            model.trainable = False

        if self.model_name == "Xception":
            model = xception.Xception(weights='imagenet', include_top=False)
            model.trainable = False

        if self.model_name == "AlexNet":
            wd = np.load("<FILEPATH TO WEIGHTS>",mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
            model = AlexNet(wd=wd, num_outputs = self.outputs, l1 = self.l1, inputs = self.inputs, pool_rate=1, output=True, prog_pool=False)

        return model

    def get_model(self):
        """get a model that consists of all the selected layers
           from the pretrained model 

        Returns:
            model.Keras: the custom model generated from the selected pretrained model
        """
        model = self.__generate_model()
        print("Model Input")
        print(model.input)

        if self.model_name != "AlexNet":
            dynamic_model = keras.Sequential([],name=self.model_name)
            dynamic_model.add(layers.Input(self.inputs))
            dynamic_model.add(self.imbedLayer)
            

            for i, layer in enumerate(model.layers):
                if i <= self.layer_number:
                    dynamic_model.add(layer)

            dynamic_model.add(layers.Flatten())
            dynamic_model.add(layers.Dense(self.outputs,trainable=True,kernel_regularizer = keras.regularizers.l1(self.l1)))
        else:
            dynamic_model = model.get_model(self.layer_number)
            
        return dynamic_model

    def model_summary(self):
        model = self.get_model()

        model.build(self.inputs)
        model.summary()


