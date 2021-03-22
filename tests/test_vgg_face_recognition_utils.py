import os
import tensorflow
import util.config as config
import numpy as np
import jsonpickle
from unittest import TestCase
import recognition.vgg_face_recognition_utils_v1 as vgg_face_utils
from recognition.ml_data import MlData

class TestPreprocessing(TestCase):

    def setUp(self):
        self.ml_data = MlData([],[],[],[], {})

    def test_serialze_ml_data(self):
        ml_data = self.ml_data
        ml_data = vgg_face_utils.load_ml_data_from_json_file(ml_data, '../recognition/data.json')
        print(ml_data)
        self.assertIsNotNone(ml_data,'ML_Data is None')

