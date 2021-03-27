import logging.config
import util.logger_init
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from keras import backend as K

class MlModel:

    def getModel(self):
        return self.model

    def summary_print(self):
        self.model.summary()


# Define TensorBoard callback child class
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)