import logging.config
import util.logger_init
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from util.tensorboard_util import plot_confusion_matrix, plot_to_image
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from keras import backend as K


class MlModel:

    def get_model(self):
        return self.model

    def summary_print(self):
        self.model.summary()

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

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define TensorBoard callback child class
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


