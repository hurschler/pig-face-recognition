import cv2
import numpy as np
import tensorflow as tf
import keras
import datetime
from keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from keras import backend as K
from tensorflow.python.keras.layers import LeakyReLU
import logging.config
import util.logger_init

from recognition.ml_model import LRTensorBoard
from recognition.ml_model import MlModel


class ClassificationModel(MlModel):

    def __init__(self, ml_data):
        self.log = logging.getLogger(__name__)
        self.log.info("init Classification Model: " + __name__)
        x_train = np.array(ml_data.x_train)
        self.model = self.define_classification_model(x_train)


    # Softmax regressor to classify images based on encoding
    def define_classification_model(self, x_train):
        # input_ = keras.layers.Input(shape=x_train.shape[1])
        # hidden1 = keras.layers.Dense(30, activation='relu')(input_)
        # hidden1 = keras.layers.BatchNormalization()(hidden1)
        # hidden1 = Dropout(0.2)(hidden1)
        # hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
        # hidden2 = keras.layers.BatchNormalization()(hidden2)
        # hidden2 = Dropout(0.2)(hidden2)
        # concat = keras.layers.Concatenate()([input_, hidden2])
        # output = keras.layers.Dense(units=10, activation='softmax')(concat)
        # classifier_model = keras.Model(inputs=[input_], outputs=[output])

        classifier_model=Sequential()
        classifier_model.add(Dense(units=1024, input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
        classifier_model.add(Activation('relu'))
        classifier_model.add(Dropout(0.2))
        classifier_model.add(Dense(units=256,kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        classifier_model.add(Activation('relu'))
        classifier_model.add(Dropout(0.2))
        classifier_model.add(Dense(units=20,kernel_initializer='he_uniform'))
        classifier_model.add(Activation('softmax'))

        # optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        optimizer=keras.optimizers.SGD(learning_rate=0.0001)
        metrics=['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        loss = loss=tf.keras.losses.SparseCategoricalCrossentropy()


        classifier_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return classifier_model


    def fit(self, ml_data):
        x_train = np.array(ml_data.x_train)
        y_train = np.array(ml_data.y_train)
        x_test = np.array(ml_data.x_test)
        y_test = np.array(ml_data.y_test)

        logdir = "../logs/recognition/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH",
                                                         circular_buffer_size=-1)

        self.checkpoint_path = '../model/face_model'

        lr_scheduler = LearningRateScheduler(self.scheduler)

        callb = [
            lr_scheduler,
            LRTensorBoard(log_dir=logdir, histogram_freq=1, write_graph=False, write_images=False, update_freq='epoch',
                          profile_batch=2, embeddings_freq=0, embeddings_metadata=None),
            ModelCheckpoint(self.checkpoint_path, save_weights_only=True, save_best_only=True, monitor="val_loss",
                            verbose=1),
        ]

        self.summary_print()
        # https://www.mt-ag.com/blog/ki-werkstatt/einstieg-in-neuronale-netze-mit-keras/ (batch_size in 2er Potenzen)
        self.model.fit(x_train, y_train, batch_size=1, epochs=100, callbacks=callb, validation_data=(x_test, y_test))

    def predict2(self, embed, left, top, right, bottom, pig_dict, img):
        width = right - left
        height = bottom - top

        img_opencv = np.array(img)
        pig = self.model.predict(embed)
        label_nr = np.argmax(pig)
        print('Accuracy score: ', pig[0][label_nr])
        # print('Accuracy score: ', pig[0][0][0][label_nr])
        print('Type of Key at dic: ', type(pig_dict.keys()))
        if label_nr in pig_dict.keys():
            print('Key found')
            name = pig_dict[label_nr]
        else:
            print('Key not found, try with string type')
            name = pig_dict[str(label_nr)]
        cv2.rectangle(img_opencv, (left, top), (right, bottom), (0, 255, 0), 2)
        img = cv2.putText(img_opencv, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                          cv2.LINE_AA)
        img = cv2.putText(img_opencv, str(np.max(pig)), (right, bottom + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 0, 255), 1, cv2.LINE_AA)

        return name

    def getModel(self):
        return self.model

    def summary_print(self):
        self.model.summary()

    def save_model(self):
        # Save model for later use
        tf.keras.models.save_model(self.model, '../model/face_classifier_model.h5')

    def load_model(self):
        # Load saved model
        self.model = tf.keras.models.load_model('../model/face_classifier_model.h5')

    # Define your scheduling function
    def scheduler(self, epoch):
        return 0.001 * 0.95 ** epoch
