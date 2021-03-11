from unittest import TestCase
from util.detection_util import DetectionUtil


class TestPreprocessing(TestCase):

    def setUp(self):
        self.detectionUtil = DetectionUtil()

    def test_getPigName(self):
        self.assertEqual("6591", self.detectionUtil.getPigName("DSC_V1_6591_2066.JPG"))


