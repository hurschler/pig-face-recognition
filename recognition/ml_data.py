from tensorflow.python.keras.callbacks_v1 import TensorBoard
from dataclasses import dataclass

@dataclass
class MlData:
    x_train: []
    y_train: []
    x_test: []
    y_test: []
    pig_dict: dict
    index_to_image_name: dict
    image_name_to_orig_image: dict
    orig_index: dict
    orig_x_train: []
    orig_y_train: []
    orig_x_test: []
    orig_y_test: []

    def last_key_value_of(self, dictionary):
        key_list = list(dictionary)
        return key_list[-1]









