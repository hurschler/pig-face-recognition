import glob
import os
import logging.config
import unittest

import recognition.config as config
from util import config
from albumentations import *
from unittest import TestCase
from augmentation import aug_util


class TestAugmentationUtil(TestCase):

    def setUp(self):
        self.log = logging.getLogger(__name__)

    # @unittest.skip("problem with local path / build server path")
    def test_save_aug_image(self):
        """Tests if the augmented image will be saved."""
        files = glob.glob(os.path.join(config.test_images_only, 'DSC*'), recursive=True)
        for file in files:
            img_orig = cv2.imread(file)
            img_cv2 = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            alpha = 1.2
            aug = RandomBrightnessContrast(p=1)
            img_aug = aug.apply(img_cv2, alpha=alpha)
            aug_util.save_aug_image(
                image_name='',
                img_path=config.test_images_only,
                pig_img_aug1=img_aug,
                prefix='GS-Test_Save-DSC_V1_6460_2238.JPG'
            )
        test_file = glob.glob(os.path.join(config.test_images_only, 'GS-Test_Save-DSC_V1_6460_2238.JPG'), recursive=True)
        self.assertIsNotNone(test_file, 'Test file is not in folder')

    def test_image_resize_height(self):
        """Tests if the height of a image will change with the resize function."""
        files = glob.glob(os.path.join(config.test_images_only, 'DSC*'), recursive=True)
        for file_name in files:
            img_orig = cv2.imread(file_name)
            height_origin = img_orig.shape[0]
            resized_image = aug_util.resize(img_orig, height=224, width=224)
            (height_new, _, _) = resized_image.shape
        self.assertNotEqual(
            height_origin,
            height_new,
            'The height has not changed. It is still ' + str(height_origin) + 'px'
        )
        self.assertEqual(
            224,
            height_new,
            'The height of the image is not 224px as expected. It is ' + str(height_new) + 'px'
        )

    def test_image_resize_width(self):
        """Tests if the width of a image will change with the resize function."""
        files = glob.glob(os.path.join(config.test_images_only, 'DSC*'), recursive=True)
        for file_name in files:
            img_orig = cv2.imread(file_name)
            width_origin = img_orig.shape[1]
            resized_image = aug_util.resize(img_orig, height=224, width=224)
            (height_new, width_new, _) = resized_image.shape
        self.assertNotEqual(
            width_origin,
            width_new,
            'The width has not changed. It is still ' + str(width_origin) + 'px'
        )
        self.assertEqual(
            224,
            width_new,
            'The width of the image is not 224px as expected. It is ' + str(width_new) + 'px'
        )

    @unittest.skip("problem with local path / build server path")
    def test_clean_augmented_images(self):
        for i, pig_name in enumerate(config.test_images_only):
            img_path = os.path.join(config.test_images_only, pig_name)
            image_names = glob.glob(os.path.join(img_path, 'DSC*'))
            for image_name in image_names:
                image_name = os.path.basename(image_name)
                img_orig = cv2.imread(os.path.join(img_path, image_name))
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                alpha = 1.2
                aug = RandomBrightnessContrast(p=1)
                pig_img_aug1 = aug.apply(img_orig, alpha=alpha)
                aug_util.save_aug_image(image_name, img_path, pig_img_aug1, 'GS-Test_Save-')
        num_of_img_before = aug_util.counts_file_in_sub_folder_with_specific_pattern(config.test_images_only, 'GS-Test_Save-DSC*')
        aug_util.clean_augmented_images(config.test_images_only, 'GS-Test_Save-*')
        num_of_img_after = aug_util.counts_file_in_sub_folder_with_specific_pattern(
            config.test_images_only,
            'GS-Test_Save-*'
        )
        self.assertTrue(num_of_img_before > num_of_img_after)

    def test_counts_file_in_sub_folder_with_specific_pattern(self):
        number_of_images_with_specific_pattern = aug_util.counts_file_in_sub_folder_with_specific_pattern(
            config.test_images_only,
            'DSC_V1_6460_223*'
        )
        self.assertTrue(number_of_images_with_specific_pattern == 1)
