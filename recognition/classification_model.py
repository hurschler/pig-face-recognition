import os

import cv2
import numpy as np
import tensorflow as tf
import datetime
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler


class ClassificationModel:

    def __init__(self, ml_data):
        print("init Classification Model")
        x_train=np.array(ml_data.x_train)
        self.model = self.define_classification_model(x_train)

    # Softmax regressor to classify images based on encoding
    def define_classification_model(self, x_train):
        classifier_model=Sequential()
        classifier_model.add(Dense(units=1024, input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
        classifier_model.add(Activation('relu'))
        classifier_model.add(Dropout(0.2))
        classifier_model.add(Dense(units=10,kernel_initializer='he_uniform'))
        classifier_model.add(Activation('softmax'))

        optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        # Bei Fehler tf.keras.metrics.AUC() anschauen oder löschen
        # BatchSize zurück setzen auf 1, Epochen 400 / 100
        metrics=['accuracy', 'mse', 'mae', 'mape', 'acc', 'categorical_accuracy', 'top_k_categorical_accuracy', tf.keras.metrics.AUC()]
        loss = loss=tf.keras.losses.SparseCategoricalCrossentropy()

        # Best Result 22.03.2021-23:11
        #classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])

        classifier_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return classifier_model

    def fit(self, ml_data):
        x_train=np.array(ml_data.x_train)
        y_train=np.array(ml_data.y_train)
        x_test=np.array(ml_data.x_test)
        y_test=np.array(ml_data.y_test)

        logdir = "../logs/recognition/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

        # self.checkpoint_path = os.path.join('model', 'test_checkpoint.h5')
        self.checkpoint_path = '../model/face_model'

        callb = [
            ModelCheckpoint(self.checkpoint_path,save_weights_only=True,save_best_only = True, monitor = "val_loss", verbose = 1),
            tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=1, embeddings_metadata=None),
        ]

        self.summary_print()
        self.model.fit(x_train, y_train,batch_size = 3, epochs=400,callbacks=callb, validation_data=(x_test, y_test))
        # self.model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
        self.summary_print()


    def predict(self, embed, rect, pig_dict, img):

        # Extract Each Face
        left = rect.rect.left()  # x1
        top = rect.rect.top()  # y1
        right = rect.rect.right()  # x2
        bottom = rect.rect.bottom()  # y2
        width = right - left
        height = bottom - top

        img_crop = img[top:top + height, left:left + width]
        cv2.imwrite(os.getcwd() + '/output/crop_img.jpg', img_crop)

        pig = self.model.predict(embed)
        name = pig_dict[np.argmax(pig)]
        # os.remove(os.getcwd() + '/crop_img.jpg')
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        img = cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, str(np.max(pig)), (right, bottom + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 0, 255), 1, cv2.LINE_AA)

        return name


    def predict2(self, embed, left, top, right, bottom, pig_dict, img):
        width = right - left
        height = bottom - top

        # img_crop = img[top:top + height, left:left + width]
        # cv2.imwrite(os.getcwd() + '/output/crop_img.jpg', img_crop)
        img_opencv = np.array(img)
        pig = self.model.predict(embed)
        label_nr = np.argmax(pig)
        print ('Accuracy score: ', pig[0][label_nr])
        print ('Type of Key at dic: ', type(pig_dict.keys()))
        if label_nr in pig_dict.keys():
            print('Key found')
            name = pig_dict[label_nr]
        else:
            print('Key not found, try with string type')
            name = pig_dict[str(label_nr)]
        cv2.rectangle(img_opencv, (left, top), (right, bottom), (0, 255, 0), 2)
        img = cv2.putText(img_opencv, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
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
        # self.model = tf.keras.models.load_model('../model/face_classifier_model.h5')
        self.model = tf.keras.models.load_model('../model/face_classifier_model.h5')

