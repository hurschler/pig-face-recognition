import logging
import os
import tensorflow
import cv2
import util.config as config
import unittest
from unittest import TestCase
from util.preprocessing import Preprocessing
from util import logger_init


class TestPreprocessing(TestCase):

    def setUp(self):
        self.preprocessing = Preprocessing()
        self.log = logging.getLogger(__name__)

    def test_compute_sharpness(self):
        """
        Tests the the preprocessing sharpness function
        Compares the function to a certain value
        """
        self.log.info('Start testing sharpness function...')
        img = self.preprocessing.readImage(config.test_images_only, 'DSC_V1_6460_2238.JPG')
        if img is None:
            img = self.preprocessing.readImage(os.path.join(
                config.build_server_path,
                'test_images_only'
            ),
                'DSC_V1_6460_2238.JPG'
            )
        self.assertIsNotNone(img, 'Image is None')
        self.assertEqual(31.764733264481336, self.preprocessing.computeSharpness(img))

    def test_replace_color(self):
        """Tests the replaceColor function form a image on a specific pixel"""
        img = self.preprocessing.readImage(
            config.test_images_only,
            'BB-DSC_V1_6494_2109.jpg'
        )
        rgb_of_pixel = img[40, 40]
        self.log.info('RGB values of the specific pixel'
                      + ' R=' + str(rgb_of_pixel[0])
                      + ' B=' + str(rgb_of_pixel[1])
                      + ' G=' + str(rgb_of_pixel[2])
                      )
        self.assertEqual(194, rgb_of_pixel[0], 'Pixel color change is not as expected')
        self.assertEqual(0, rgb_of_pixel[1], 'Pixel color change is not as expected')
        self.assertEqual(0, rgb_of_pixel[2], 'Pixel color change is not as expected')
        img = self.preprocessing.replaceColor(img, 0, 0, 194)
        img = self.preprocessing.replaceColor(img, 5, 6, 150)
        rgb_of_pixel = img[45, 40]
        self.log.info('RGB values of the specific pixel'
                      + ' R=' + str(rgb_of_pixel[0])
                      + ' B=' + str(rgb_of_pixel[1])
                      + ' G=' + str(rgb_of_pixel[2])
                      )
        self.assertEqual(0, rgb_of_pixel[0], 'Pixel color change is not as expected')
        self.assertEqual(0, rgb_of_pixel[1], 'Pixel color change is not as expected')
        self.assertEqual(0, rgb_of_pixel[2], 'Pixel color change is not as expected')
        cv2.imwrite(os.path.join(config.test_images_only, 'RBB-DSC_V1_6494_2109.jpg'), img)

