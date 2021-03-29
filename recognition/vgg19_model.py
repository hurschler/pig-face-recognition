from random import random

import tensorflow as tf
import logging.config
import datetime
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import layers
from tensorflow.python.ops.init_ops_v2 import glorot_uniform
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
import os
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import logging.config
import util.logger_init
import util.logger_init


# Wrapper Class around Keras Model
class Vgg19:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init Vgg19")
        self.height = 224
        self.width = 224
        self.model = self.define_model()

    def define_model(self):
        inputs = layers.Input(shape=(self.width, self.height, 3))
        model = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            pooling='avg',
            classifier_activation="softmax",
        )
        # Freeze the pretrained weights
        model.trainable = False
        x = model.output
        model = tf.keras.Model(inputs, outputs=x, name="Vgg19")
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        metrics = ['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def getModel(self):
        return self.model

    def get_embeddings(self, crop_img_name):
        # Get Embeddings
        crop_img = load_img(crop_img_name, target_size=(self.width, self.height))
        crop_img = img_to_array(crop_img)
        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = preprocess_input(crop_img)
        return self.model(crop_img)

    def getEmbeddings(self, img):
        return self.model(img)

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height
