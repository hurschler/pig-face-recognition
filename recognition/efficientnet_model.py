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
# https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
class EfficientNetModel:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init EfficientNetModel")
        self.model = self.define_model()



    def define_model(self):
        # inputs = layers.Input(shape=(224, 224, 3))
        inputs = layers.Input(shape=(600, 600, 3))
        # model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
        model = EfficientNetB7(include_top=False, input_tensor=inputs, weights="imagenet")

        # Freeze the pretrained weights
        model.trainable = False

        x = model.output
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        # x = tf.keras.layers.Dense(1,activation='sigmoid')(x)

        # outputs = layers.Dense(1, activation="softmax", name="pred")(x)
        # outputs = layers.Dense(10, activation="softmax", name="pred")(x)

        model = tf.keras.Model(inputs, outputs=x, name="EfficientNet")
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
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
        self.model.fit_generator(train_generator, epochs=70, callbacks=callb, validation_data=validation_generator)



    def load_weights(self):
        # self.sequential_model.load_weights('../model/vgg_face_weights.h5')$
        # Todo efficientnet load_weights
        print('Todo efficientnet load_weights')


    def save_weights(self):
        self.model.save_weights('../model/efficientnet.h5')

    def save_model(self):
        # Save model for later use
        tf.keras.models.save_model(self.model, '../model/efficientnet_face_model.h5')

    def load_model(self):
        # Load saved model
        self.model = tf.keras.models.load_model('../model/efficientnet_face_model.h5')

    def getModel(self):
        return self.model


    def get_embeddings(self, crop_img_name):
        # Get Embeddings
        crop_img = load_img(crop_img_name, target_size=(600, 600))
        crop_img = img_to_array(crop_img)
        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = preprocess_input(crop_img)
        return self.model(crop_img)

    def efficientnet_face(self, img):
        return self.model(img)