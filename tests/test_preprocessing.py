import tensorflow
import util.config as config
from unittest import TestCase
from util.preprocessing import Preprocessing


class TestPreprocessing(TestCase):

    def setUp(self):
        self.preprocessing = Preprocessing()

    def test_compute_sharpness(self):
        img = self.preprocessing.readImage(config.image_sample_path, config.image_example_name)
        self.assertEqual(31.764733264481336, self.preprocessing.computeSharpness(img))

