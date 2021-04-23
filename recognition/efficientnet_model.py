import tensorflow as tf
import logging.config
import datetime
import logging.config
import numpy as np
import os
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from random import random
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.ops.init_ops_v2 import glorot_uniform
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from matplotlib import pyplot
from tensorflow.keras.models import Sequential, Model


# https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
from recognition.feature_extraction_model import FeatureExtractionModel


class EfficientNetModel(FeatureExtractionModel):
    """
    Wrapper Class around Keras Model
    It defines the EfficientNetB7
    """

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("Init EfficientNetModel")
        self.model = self.define_model()

    def define_model(self):
        """Defines the model of the EfficientNetB7"""
        self.log.info('Defining EfficientNetB7...')
        inputs = layers.Input((600, 600, 3))
        model = EfficientNetB7(
            include_top=False,
            input_tensor=inputs,
            weights="imagenet"
        )
        model.trainable = True
        x = model.output
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        # x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs=x, name="EfficientNet")
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        metrics = ['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def fit(self, train_generator, validation_generator):
        """
        Trains the efficientNetB7 with train generator
        @Params:
            - train_generator:
            - validation_generator:
        """
        logdir = "../logs/recognition/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tf.debugging.experimental.enable_dump_debug_info(
            logdir,
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1
        )
        self.checkpoint_path = '../model/face_model'
        callb = [
            TensorBoard(
                log_dir=logdir,
                histogram_freq=0,
                write_graph=False,
                write_images=True,
                update_freq='epoch',
                profile_batch=2,
                embeddings_freq=0,
                embeddings_metadata=None
            ),
            ModelCheckpoint(
                self.checkpoint_path,
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
                verbose=1
            ),
        ]
        self.model.summary()
        self.model.fit_generator(
            train_generator,
            epochs=70,
            callbacks=callb,
            validation_data=validation_generator
        )

    def load_weights(self):
        # self.sequential_model.load_weights('../model/vgg_face_weights.h5')$
        # Todo efficientNet load_weights
        print('Todo efficientnet load_weights')

    def save_weights(self):
        """Saves the weights in the model folder"""
        self.log.info('Saving weights...')
        self.model.save_weights('../model/efficient_net_b7.h5')

    def get_embeddings(self, img):
        """Returns model efficient_net with loaded image"""
        self.log.info('Getting model...')
        return self.model(img)