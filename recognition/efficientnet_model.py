from random import random

import tensorflow as tf
import logging.config
import datetime
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import layers
from tensorflow.python.ops.init_ops_v2 import glorot_uniform

import util.logger_init


# Wrapper Class around Keras Model
# https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
class EfficientNetModel:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init EfficientNetModel")
        self.model = self.define_model()



    def define_model(self):
        inputs = layers.Input(shape=(224, 224, 3))
        model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
        # Freeze the pretrained weights
        model.trainable = False

        x = model.output
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2, name="top_dropout")(x)
        x = layers.Dense(1280, activation='relu',  kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
        # x = layers.BatchNormalization()(x)

        # outputs = layers.Dense(1, activation="softmax", name="pred")(x)
        outputs = layers.Dense(10, activation="softmax", name="pred")(x)

        model = tf.keras.Model(inputs, outputs, name="EfficientNet")
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
        self.model.fit_generator(train_generator, epochs=100, callbacks=callb, validation_data=validation_generator)



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