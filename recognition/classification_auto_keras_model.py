import cv2
import numpy as np
import tensorflow as tf
import keras
import datetime
import autokeras as ak
from keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from keras import backend as K
from tensorflow.python.keras.layers import LeakyReLU
from recognition.ml_model import MlModel
import logging.config
import util.logger_init


class ClassificationAutoKerasModel(MlModel):

    def __init__(self, ml_data):
        self.log = logging.getLogger(__name__)
        self.log.info("init Classification Model: " + __name__)
        x_train = np.array(ml_data.x_train)
        self.model = self.define_classification_model(ml_data.x_train)


    # Softmax regressor to classify images based on encoding
    def define_classification_model(self, x_train):
        clf = ak.StructuredDataClassifier(overwrite=True, max_trials=10)
        return clf

    def fit(self, ml_data):
        x_train = np.array(ml_data.x_train)
        y_train = np.array(ml_data.y_train)
        x_test = np.array(ml_data.x_test)
        y_test = np.array(ml_data.y_test)

        logdir = "../logs/recognition/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callb = [
            LRTensorBoard(log_dir=logdir, histogram_freq=1, write_graph=False, write_images=False, update_freq='epoch',
                          profile_batch=2, embeddings_freq=0, embeddings_metadata=None),
        ]

        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, callbacks=callb)
        print ("evaluate: ", self.model.evaluate(x_test, y_test))

    def predict2(self, embed, left, top, right, bottom, pig_dict, img):
        width = right - left
        height = bottom - top

        img_opencv = np.array(img)
        pig = self.model.predict(embed)
        label_nr = np.argmax(pig)
        print('Accuracy score: ', pig[0][label_nr])
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
        exp_model = self.model.export_model()
        exp_model.save('../model/model_autokeras', save_format='tf')

    def load_model(self):
        self.model = load_model('../model/model_autokeras', custom_objects=ak.CUSTOM_OBJECTS )
        self.log.info(self.model.summary())



# Define TensorBoard callback child class
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

