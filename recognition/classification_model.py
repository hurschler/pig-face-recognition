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

    def __init__(self, ml_data, number_of_pigs):
        self.logdir = "../logs/recognition/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer_cm = tf.summary.create_file_writer(self.logdir + '/cm')
        self.checkpoint_path = '../model/face_model'
        self.log = logging.getLogger(__name__)
        self.log.info("Init Classification Model: " + __name__)
        x_train = np.array(ml_data.x_train)
        self.number_of_pigs = number_of_pigs
        self.model = self.define_classification_model(x_train, number_of_pigs)
        self.ml_data = ml_data

    def define_classification_model(self, x_train, number_of_pigs):
        """Softmax regressor to classify images based on encoding"""
        kernel_init = keras.initializers.lecun_normal()
        classifier_model = Sequential()
        classifier_model.add(
            Dense(
                units=32,
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                input_dim=x_train.shape[1],
                kernel_initializer=kernel_init
            )
        )
        classifier_model.add(BatchNormalization())
        classifier_model.add(Activation('relu'))
        classifier_model.add(Dropout(0.2))
        classifier_model.add(Dense(
            units=32,
            kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            kernel_initializer=kernel_init
        ))
        classifier_model.add(Activation('relu'))
        classifier_model.add(Dropout(0.2))
        classifier_model.add(Dense(
            units=number_of_pigs,
            kernel_initializer=kernel_init
        ))
        classifier_model.add(Activation('softmax'))
        optimizer = keras.optimizers.Nadam(
            lr=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            schedule_decay=0.004
        )
        metrics = ['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        classifier_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return classifier_model

    def fit(self, ml_data, batch_size, epochs):
        x_train = np.array(ml_data.x_train)
        y_train = np.array(ml_data.y_train)
        x_test = np.array(ml_data.x_test)
        y_test = np.array(ml_data.y_test)
        tf.debugging.experimental.enable_dump_debug_info(
            self.logdir,
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1
        )
        lr_scheduler = LearningRateScheduler(self.scheduler)
        cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=self.log_confusion_matrix)

        callb = [
            cm_callback,
            lr_scheduler,
            LRTensorBoard(
                log_dir=self.logdir,
                histogram_freq=1,
                write_graph=False,
                write_images=False,
                update_freq='epoch',
                profile_batch=2,
                embeddings_freq=0,
                embeddings_metadata=None
            ),
            ModelCheckpoint(
                self.checkpoint_path,
                save_weights_only=True,
                save_best_only=True,
                monitor="val_accuracy",
                mode='max',
                verbose=1),
        ]
        self.summary_print()
        self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callb,
            validation_data=(x_test, y_test)
        )


    def fit_with_k_fold(self, ml_data, batch_size, epochs, k):
        # Merge inputs and targets
        x_train = np.array(ml_data.x_train)
        y_train = np.array(ml_data.y_train)
        x_test = np.array(ml_data.x_test)
        y_test = np.array(ml_data.y_test)
        inputs = np.concatenate((x_train, x_test), axis=0)
        targets = np.concatenate((y_train, y_test), axis=0)
        kfold = KFold(n_splits=k, shuffle=True)
        lr_scheduler = LearningRateScheduler(self.scheduler)
        cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=self.log_confusion_matrix)

        tf.debugging.experimental.enable_dump_debug_info(
            self.logdir,
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1
        )
        callb = [
            cm_callback,
            lr_scheduler,
            LRTensorBoard(
                log_dir=self.logdir,
                histogram_freq=1,
                write_graph=False,
                write_images=False,
                update_freq='epoch',
                profile_batch=2,
                embeddings_freq=0,
                embeddings_metadata=None
            ),
            ModelCheckpoint(
                self.checkpoint_path,
                save_weights_only=True,
                save_best_only=True,
                monitor="val_accuracy",
                mode='max',
                verbose=1
            ),
        ]
        self.summary_print()

        fold_no = 1
        acc_per_fold = []
        loss_per_fold = []
        for train, test in kfold.split(inputs, targets):
            self.model = self.define_classification_model(x_train, self.number_of_pigs)
            history = self.model.fit(
                inputs[train],
                targets[train],
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callb,
                validation_data=(x_test, y_test))
            scores = self.model.evaluate(inputs[test], targets[test], verbose=1)
            self.log.info('Score for fold ' + str(fold_no) + ' ' + self.model.metrics_names[0] + ' of ' +
                          str(scores[0]) + ' ' + self.model.metrics_names[1] + ' of ' + str(scores[1]*100) + '%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            fold_no = fold_no + 1

        self.log.info('-------------------------------------------------------------------------------------')
        self.log.info('Avarage scores for all folds: ')
        self.log.info('Accuracy: ' + str(np.mean(acc_per_fold)) + ' Std: ' + str(np.std(acc_per_fold)))
        self.log.info('Loss: ' + str(np.mean(loss_per_fold)))
        self.log.info('-------------------------------------------------------------------------------------')







    def predict(self, embed, left, top, right, bottom, pig_dict, img):
        width = right - left
        height = bottom - top
        img_opencv = np.array(img)
        pig = self.model.predict(embed)
        label_nr = np.argmax(pig)
        self.log.debug('Accuracy score: ', pig[0][label_nr])
        self.log.debug('Type of Key at dic: ', type(pig_dict.keys()))
        if label_nr in pig_dict.keys():
            print('Key found')
            name = pig_dict[label_nr]
        else:
            print('Key not found, try with string type')
            name = pig_dict[str(label_nr)]
        cv2.rectangle(img_opencv, (left, top), (right, bottom), (0, 255, 0), 2)
        return name

    def predict_label(self, embed):
        pig = self.model.predict(embed)
        label_nr = np.argmax(pig)
        print('Accuracy score: ', pig[0][label_nr])
        return label_nr

    def get_model(self):
        return self.model

    def summary_print(self):
        self.model.summary()

    def save_model(self):
        """Saves the model for later use"""
        self.log.info('Saving the model...')
        tf.keras.models.save_model(self.model, '../model/face_classifier_model.h5')

    def load_model(self):
        """Loads saved model"""
        self.log.info('Loading model...')
        self.model = tf.keras.models.load_model('../model/face_classifier_model.h5')

    def scheduler(self, epoch):
        """Calculates the learning rate"""
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
