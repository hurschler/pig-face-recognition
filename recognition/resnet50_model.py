from random import random

import tensorflow as tf
import logging.config
import datetime
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import layers
from tensorflow.python.keras.layers import AveragePooling2D
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


# Wrapper Class around Keras Model

class ResNetModel:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init ResNetModel")
        self.model = self.define_model()



    def define_model(self):
        inputs = layers.Input(shape=(224, 224, 3))
        # model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
        # model = tf.keras.applications.ResNet152V2(
        model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=(224, 224, 3)
        )

        x = model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        # Patrick & André (Zeile entfernen)
        output_layer = Dense(20, activation='softmax', name='softmax')(x)


        model = tf.keras.Model(model.inputs, outputs=output_layer, name="ResNet50")

        for layer in [l for l in model.layers if 'conv5' not in l.name]:
            layer.trainable = False

        for layer in [l for l in model.layers if 'conv5' in l.name or l.name == 'probs']:
            layer.trainable = True

        # x = model.output
        # x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        metrics = ['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
        self.log.info(model.summary())
        return model


    def fit(self, train_generator, validation_generator):
        logdir = "../logs/recognition/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH",
                                                         circular_buffer_size=-1)
        self.checkpoint_path = '../model/face_model'

        callb = [
            TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=False, write_images=True, update_freq='epoch',
                          profile_batch=2, embeddings_freq=0, embeddings_metadata=None),
            ModelCheckpoint(self.checkpoint_path, save_weights_only=True, save_best_only=True, monitor="val_loss",
                            verbose=1),
        ]

        self.model.summary()
        self.model.fit_generator(train_generator, epochs=10, callbacks= callb,  validation_data=validation_generator)

    def get_model(self):
        return self.model

    def load_weights(self):
        self.log.info('Loading weights...')
        self.model.load_weights('../model/resnet50_v2.h5')

    def save_weights(self):
        self.log.info('Saving weights...')
        self.model.save_weights('../model/resnet50_v2.h5')

    def getResnet_face(self, img):
        return self.model(img)