import os
import tensorflow
import util.config as config
import numpy as np
import jsonpickle
from unittest import TestCase
from recognition.ml_data import MlData


class TestPreprocessing(TestCase):

    def setUp(self):
        self.ml_data = MlData([],[],[],[], {})

    def test_serialze_ml_data(self):
        ml_data = self.ml_data
        ml_data.x_test
        ml_data.pig_dict[0] = '6440'
        ml_data.x_train.append(np.array([1, 2, 3]).tolist())
        ml_data.y_train.append(0)
        ml_data_json = jsonpickle.encode(ml_data)
        print(ml_data_json)
        self.assertIsNotNone(ml_data_json,'JSON is None')

