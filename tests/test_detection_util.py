from unittest import TestCase
from util.detection_util import DetectionUtil


class TestPreprocessing(TestCase):

    def setUp(self):
        self.detectionUtil = DetectionUtil()
        self.image_name_1 = "DSC_V1_6591_2066.JPG"
        self.image_name_2 = "DSC_V2_6591_2066.JPG"

    def test_getPigName(self):
        self.assertEqual("6591", self.detectionUtil.getPigName(self.image_name_1))


    def test_setVersion(self):
        self.assertEqual("V1", self.detectionUtil.getSetVersion(self.image_name_1))
        self.assertNotEqual("V2", self.detectionUtil.getSetVersion(self.image_name_1))
        self.assertEqual("V2", self.detectionUtil.getSetVersion(self.image_name_2))

