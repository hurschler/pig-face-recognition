import cv2
import recognition.config as config
from unittest import TestCase
import logging.config


class TestPreprocessing(TestCase):

    def setUp(self):
        """Sets up the logger and it's configuration"""
        self.log = logging.getLogger(__name__)

    def test_read_config(self):
        """Tests the reading of the configuration paths"""
        image_root_dir_path = config.image_root_dir_path
        self.log.info(image_root_dir_path)
        self.assertIsNotNone(image_root_dir_path)

    def test_read_sample_image(self):
        """
        Tests the sample folder if a specific file exists
        Tests implicit function imread()
        """
        sample_img_full_path = config.image_sample_full_path
        self.log.info(sample_img_full_path)
        self.assertIsNotNone(sample_img_full_path, 'Path is empty: ' + str(sample_img_full_path))
        img = cv2.imread(sample_img_full_path)
        self.assertIsNotNone(img, 'image not readable:' + str(sample_img_full_path))

    def test_read_integer(self):
        """Tests if typ int is explicit typ int"""
        i = config.keras_max_augmentation
        self.assertTrue(isinstance(i, int), 'Type ist not a Int')
        self.log.info(isinstance(i, int))
        self.log.info('Actual typ: ' + str(type(i)))

    def test_read_path_from_build_properties(self):
        """Tests specific path from build.properties"""
        build_server_path = config.build_server_path
        self.log.info('Received path: ' + str(build_server_path))
        self.assertIsNotNone(build_server_path)
