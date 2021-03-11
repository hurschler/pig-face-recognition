import tensorflow
from unittest import TestCase
from util.preprocessing import Preprocessing


class TestPreprocessing(TestCase):

    def setUp(self):
        self.preprocessing = Preprocessing()

    def test_define_model(self):
        # self.assertNotEqual(self.vgg_face_model, None)
        self.preprocessing.readImages()

    def test_compute_sharpness(self):
        img_dic = self.preprocessing.readImages()
        self.preprocessing.computeSharpness(img_dic)

