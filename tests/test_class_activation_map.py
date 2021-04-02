import os
import unittest

import tensorflow
import util.config as config
import numpy as np
import jsonpickle
import util.class_activation_map
from unittest import TestCase
from util import class_activation_map


@unittest.skip('file not found')
class TestPreprocessing(TestCase):

    def setUp(self):
        print('Image full path: ', os.path.join(config.image_sample_path, config.image_example_name))
        img = self.preprocessing.readImage(config.image_sample_path, config.image_example_name)
        if img is None:
            img = self.preprocessing.readImage(os.path.join(config.build_server_path, 'sample'),
                                               config.image_example_name)

    def test_generate_class_activation_map(self):
        class_activation_map.generate_class_activation_map()
