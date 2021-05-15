import os
import tensorflow
import util.config as config
import recognition.ml_util
import numpy as np
import jsonpickle
from unittest import TestCase

from recognition import ml_util
from recognition.ml_data import MlData



class TestMlUtil(TestCase):

    def setUp(self):
        self.ml_data = MlData([],[],[],[], {}, {}, {}, {}, [],[],[],[])

    def test_check_is_augmented_image_name(self):
        not_aug_image_name = 'DSC_V2_6408_2593.JPG-crop-mask0.jpg'
        self.assertFalse(ml_util.check_is_augmented_image_name(not_aug_image_name))
        aug_image_name = '0-DSC_V2_6408_2593.JPG-crop-mask0.jpg'
        self.assertTrue(ml_util.check_is_augmented_image_name(aug_image_name))

        ml_data = self.ml_data
        ml_data.x_test
        ml_data.pig_dict[0] = '6440'
        ml_data.x_train.append(np.array([1, 2, 3]).tolist())
        ml_data.y_train.append(0)
        ml_data_json = jsonpickle.encode(ml_data)
        print(ml_data_json)
        self.assertIsNotNone(ml_data_json,'JSON is None')

