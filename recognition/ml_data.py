from tensorflow.python.keras.callbacks_v1 import TensorBoard
from dataclasses import dataclass

@dataclass
class MlData:
    x_train: []
    y_train: []
    x_test: []
    y_test: []
    pig_dict: dict


# Define TensorBoard callback child class
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)




