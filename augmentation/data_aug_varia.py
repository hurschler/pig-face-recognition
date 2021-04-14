import os
import glob
from albumentations import RandomBrightnessContrast, CLAHE, ToGray, Blur, RandomFog, HueSaturationValue, cv2
from matplotlib import pyplot as plt
import logging.config
from augmentation.aug_util import save_aug_image, clean_augmented_images
from util import config

log = logging.getLogger(__name__)
PATH_TO_IMAGES = config.output_path_cropped_rectangle


# https://github.com/albumentations-team/albumentations#installation
def generate_aug_images(path=PATH_TO_IMAGES):
    """
    Generates augmented images of a specific folder
    Augmentations:
        - Randomly change brightness and contrast of the input image
        - Apply Contrast Limited Adaptive Histogram Equalization to the input image
        - Convert the input RGB image to grayscale
        - Blur the input image using a random-sized kernel
        - Simulates fog for the image
        - Randomly change hue, saturation and value of the input image
    """
    log.info('Generating augmentation of the images...')
    log.info('Path of the images: ' + PATH_TO_IMAGES)
    for i, pig_name in enumerate(os.listdir(path)):
        img_path = os.path.join(path, pig_name)
        image_names = glob.glob(os.path.join(img_path, 'DSC*'))
        for image_name in image_names:
            image_name = os.path.basename(image_name)
            img_orig = cv2.imread(os.path.join(img_path, image_name))
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

            alpha = 1.2
            aug = RandomBrightnessContrast(p=1)
            pig_img_aug1 = aug.apply(img_orig, alpha=alpha)
            save_aug_image(image_name, img_path, pig_img_aug1, 'A1-')

            aug = CLAHE(p=1.0)
            pig_img_aug2 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug2, 'A2-')

            aug = ToGray(p=0.5)
            pig_img_aug3 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug3, 'A3-')

            aug = Blur(p=0.5, blur_limit=7)
            pig_img_aug4 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug4, 'A4-')

            aug = RandomFog(p=1, fog_coef_lower=0.1, fog_coef_upper=0.1, alpha_coef=0.8)
            pig_img_aug5 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug5, 'A5-')

            aug = HueSaturationValue(hue_shift_limit=200, sat_shift_limit=70, val_shift_limit=27, p=1)
            pig_img_aug6 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug6, 'A6-')

        print("augmentation in process A1: " + str(i))
    print('augmentation finished (sharpness)')


# -----------------------------------------------------------------------------------------------------------

# generate_aug_images()
clean_augmented_images(path=PATH_TO_IMAGES)

