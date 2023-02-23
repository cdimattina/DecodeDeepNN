# ==========================================================
# File      : dynamic_pretrained_model.py
# Author    : Josiah Burnham (2-23-2023)
# Purpose   : implementation of the DynamicPretrainedModel Class
# ==========================================================

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications import *


class DynamicPretrainedModel:
    def __init__(self, model_name, layer_number, inputs, outputs):
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

    def __generate_model(self):
        """ download the pretrained model to get the individual layer
            activation of

        Returns:
            keras.Model: the pretrain model to get the layers of
        """
        if self.model_name == "VGG16":
            model = vgg16.VGG16(weights='imagenet', include_top=False)

        if self.model_name == "VGG19":
            model = vgg19.VGG19(weights='imagenet', include_top=False)

        if self.model_name == "ResNet50":
            model = resnet50.ResNet50(weights='imagenet', include_top=False)

        if self.model_name == "InceptionV3":
            model = inception_v3.InceptionV3(
                weights='imagenet', include_top=False)

        if self.model_name == "Xception":
            model = xception.Xception(weights='imagenet', include_top=False)

        model.trainable = False

        return model

    def get_model(self):
        """get a model that consists of all the selected layers
           from the pretrained model 

        Returns:
            model.Keras: the custom model generted from the selected pretrained model
        """
        model = self.__generate_model()
        print("Model Input")
        print(model.input)

        dynamic_model = keras.Sequential(name=self.model_name)
        dynamic_model.add(layers.Input(self.inputs))

        for i, layer in enumerate(model.layers):
            if i <= self.layer_number:
                dynamic_model.add(layer)

        dynamic_model.add(layers.Flatten())
        dynamic_model.add(layers.Dense(self.outputs))

        return dynamic_model

    def model_summary(self):
        model = self.get_model()

        model.build(self.inputs)
        model.summary()


dynamic = DynamicPretrainedModel("Xception", 6, (60, 60, 3), 2)
dynamic.model_summary()
