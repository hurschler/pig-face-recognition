import util.config as config
import recognition.vgg_face_recognition_utils_v1 as vgg_face_utils
from unittest import TestCase
from recognition.ml_data import MlData

PATH_JSON_DATA = config.test_json


class TestPreprocessing(TestCase):

    def setUp(self):
        self.ml_data = MlData([], [], [], [], {})

    def test_serialize_ml_data(self):
        """Tests the serialisation of ml_Data"""
        ml_data = self.ml_data
        ml_data = vgg_face_utils.load_ml_data_from_json_file(
            ml_data,
            json_data=PATH_JSON_DATA
        )
        self.assertIsNotNone(ml_data, 'ML_Data is None')
