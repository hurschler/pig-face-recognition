import cv2
import numpy as np
# import sklearn
import tensorflow as tf
import keras
import datetime
from sklearn.metrics import confusion_matrix
from keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from keras import backend as K
from tensorflow.python.keras.layers import LeakyReLU, Normalization, LayerNormalization, PReLU
import logging.config
import util.logger_init
import util.performance_visualization_callback as perfvis
import util.tensorboard_util as tbutil
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import matplotlib.pyplot as plt

from recognition.ml_model import LRTensorBoard
from recognition.ml_model import MlModel
from util.tensorboard_util import plot_confusion_matrix, plot_to_image


class ClassificationModel(MlModel):

    def __init__(self, ml_data):
        self.logdir = "../logs/recognition/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer_cm = tf.summary.create_file_writer(self.logdir + '/cm')
        self.checkpoint_path = '../model/face_model'
        self.log = logging.getLogger(__name__)
        self.log.info("Init Classification Model: " + __name__)
        x_train = np.array(ml_data.x_train)
        self.model = self.define_classification_model(x_train)
        self.ml_data = ml_data

    # Softmax regressor to classify images based on encoding
    def define_classification_model(self, x_train):
        # stabile 0.375 - 0.4 auf vgg16 (lecun_normal)
        kernel_init = keras.initializers.lecun_normal()
        # kernel_init = keras.initializers.glorot_normal()

        classifier_model = Sequential()
        classifier_model.add(
            Dense(units=32, kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4), input_dim=x_train.shape[1],
                  kernel_initializer=kernel_init))
        classifier_model.add(BatchNormalization())
        classifier_model.add(Activation('relu'))
        classifier_model.add(Dropout(0.2))
        classifier_model.add(Dense(units=32, kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                   kernel_initializer=kernel_init))
        classifier_model.add(Activation('relu'))
        classifier_model.add(Dropout(0.2))
        classifier_model.add(Dense(units=20, kernel_initializer=kernel_init))
        classifier_model.add(Activation('softmax'))

        optimizer = keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        # optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # optimizer = keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
        # optimizer=keras.optimizers.SGD(learning_rate=0.0001)
        metrics = ['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        classifier_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return classifier_model

    def fit(self, ml_data):
        x_train = np.array(ml_data.x_train)
        y_train = np.array(ml_data.y_train)
        x_test = np.array(ml_data.x_test)
        y_test = np.array(ml_data.y_test)

        tf.debugging.experimental.enable_dump_debug_info(self.logdir, tensor_debug_mode="FULL_HEALTH",
                                                         circular_buffer_size=-1)
        lr_scheduler = LearningRateScheduler(self.scheduler)
        cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=self.log_confusion_matrix)

        callb = [
            cm_callback,
            lr_scheduler,
            LRTensorBoard(log_dir=self.logdir, histogram_freq=1, write_graph=False, write_images=False,
                          update_freq='epoch',
                          profile_batch=2, embeddings_freq=0, embeddings_metadata=None),
            ModelCheckpoint(self.checkpoint_path, save_weights_only=True, save_best_only=True, monitor="val_accuracy",
                            mode='max',
                            verbose=1),
        ]

        self.summary_print()
        # https://www.mt-ag.com/blog/ki-werkstatt/einstieg-in-neuronale-netze-mit-keras/ (batch_size in 2er Potenzen)
        # self.model.fit(x_train, y_train, batch_size=25, epochs=200, callbacks=callb, validation_data=(x_test, y_test))
        self.model.fit(x_train, y_train, batch_size=45, epochs=200, callbacks=callb, validation_data=(x_test, y_test))

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

    def predict_label(self, embed):
        pig = self.model.predict(embed)
        label_nr = np.argmax(pig)
        print('Accuracy score: ', pig[0][label_nr])
        return label_nr

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

    def log_confusion_matrix(self, epoch, logs):

        # Use the model to predict the values from the test_images.
        test_pred_raw = self.model.predict(self.ml_data.x_test)

        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix using sklearn.metrics
        cm = confusion_matrix(self.ml_data.y_test, test_pred)

        figure = plot_confusion_matrix(cm, class_names=self.ml_data.pig_dict.values())
        cm_image = plot_to_image(figure)

        # plot and save roc curve
        x_test = np.array(self.ml_data.x_test)
        y_test = np.array(self.ml_data.y_test)
        y_pred = np.asarray(self.model.predict((x_test, y_test)[0]))
        y_true = (x_test, y_test)[1]
        y_pred_class = np.argmax(y_pred, axis=1)
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_true, y_pred, classes_to_plot=[0, 'cold'], ax=ax)
        roc_img = plot_to_image(fig)

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
            tf.summary.image("ROC Curve", roc_img, step=epoch)
