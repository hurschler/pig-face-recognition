import os
import tensorflow
import cv2
import util.config as config
from unittest import TestCase
from util.preprocessing import Preprocessing


class TestPreprocessing(TestCase):

    def setUp(self):
        self.preprocessing = Preprocessing()

    def test_compute_sharpness(self):
        print ('Image full path: ', os.path.join(config.image_sample_path, config.image_example_name))
        img = self.preprocessing.readImage(config.image_sample_path, config.image_example_name)
        if img is None:
            img = self.preprocessing.readImage(os.path.join(config.build_server_path,'sample'), config.image_example_name)
        self.assertIsNotNone(img,'Image is None')
        self.assertEqual(31.764733264481336, self.preprocessing.computeSharpness(img))

    def test_replace_color(self):
        print ('Image full path: ', os.path.join(config.image_sample_path, config.image_example_blue_name))
        img = self.preprocessing.readImage(config.image_sample_path, config.image_example_blue_name)
        img = self.preprocessing.replaceColor(img, 0, 0, 194)
        img = self.preprocessing.replaceColor(img, 5, 6, 150)
        cv2.imwrite(os.path.join(config.output_path, 'color_replaced.jpg'), img)
