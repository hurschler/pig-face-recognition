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
class NasNetLarge:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init NasNetLarge")
        self.model = self.define_model()



    def define_model(self):
        inputs = layers.Input(shape=(331, 331, 3))
        model = tf.keras.applications.NASNetLarge(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            # pooling='avg',
            pooling='avg',
        )
        # Freeze the pretrained weights
        model.trainable = False
        x = model.output
        model = tf.keras.Model(inputs, outputs=x, name="NasNetLarge")
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        metrics = ['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
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
        self.model.fit_generator(train_generator, epochs=50, callbacks=callb, validation_data=validation_generator)


    def save_model(self):
        # Save model for later use
        tf.keras.models.save_model(self.model, '../model/NasNetLarge-model.h5')

    def load_model(self):
        # Load saved model
        self.model = tf.keras.models.load_model('../model/NasNetLarge-model.h5')

    def getModel(self):
        return self.model


    def get_embeddings(self, crop_img_name):
        # Get Embeddings
        crop_img = load_img(crop_img_name, target_size=(331, 331))
        crop_img = img_to_array(crop_img)
        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = preprocess_input(crop_img)
        return self.model(crop_img)

    def efficientnet_face(self, img):
        return self.model(img)